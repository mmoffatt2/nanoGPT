import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from quantization.quantize import _fake_quantize, quantize_dictionary

class BitLinear1p58(nn.Linear):
    """ BitLinear from Era of 1.58 LLMs Paper
    Source: https://huggingface.co/1bitLLM/bitnet_b1_58-large/blob/main/utils_quant.py
    Source License: MIT
    Paper Link: https://arxiv.org/abs/2402.17764
    """

    def __init__(self, in_features, out_features, config, num_groups=1):
        super().__init__(in_features, out_features, config.bias)

        """
        RMSNorm is placed outside BitLinear
        """
        weight_bits=1
        input_bits=8
        self.weight_bits = weight_bits
        self.input_bits = input_bits

    def forward(self, x):

        quant_input = x + (self.activation_quant(x, self.input_bits) - x).detach()
        quant_weight = self.weight + (self.weight_quant(self.weight, self.weight_bits) - self.weight).detach()

        out = nn.functional.linear(quant_input, quant_weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def weight_quant(self, weight, num_bits=1):
        dtype = weight.dtype
        weight = weight.float()
        s =  1 / weight.abs().mean().clamp(min=1e-5)
        result = (weight * s).round().clamp(-1, 1) / s
        return result.type(dtype)

    def activation_quant(self, x, num_bits=8):
        dtype = x.dtype
        x = x.float()
        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1
        s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (x * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

class BitLinear(nn.Linear):
    """PyTorch BitLinear Layer
    Source: https://github.com/Beomi/BitNet-Transformers/tree/main
    Source License: Apache Version 2.0
    """

    def __init__(self, in_features, out_features, config, num_groups=1):
        super(BitLinear, self).__init__(in_features, out_features, config.bias)
        self.num_groups = num_groups
        self.eps = 1e-5

    def ste_binarize(self, x):
        # Apply the sign function for binarization
        binarized_x = torch.sign(x)
        # Use STE: during backward pass, we bypass the binarization
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def binarize_weights_groupwise(self):
        # Divide weights into groups
        group_size = self.weight.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(self.weight)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[start_idx:end_idx]

            # Binarize each group using STE
            alpha_g = weight_group.mean()
            binarized_weights[start_idx:end_idx] = self.ste_binarize(
                weight_group - alpha_g
            )

        return binarized_weights

    def quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, input):
        # Binarize weights (group-wise) using STE
        binarized_weights = self.binarize_weights_groupwise()

        # Normal linear transformation with binarized weights
        output = torch.nn.functional.linear(input, binarized_weights, self.bias)

        # Quantize activations group-wise
        output = self.quantize_activations_groupwise(output)

        return output


class BitLinearOptimized(nn.Linear):
    """Memory Optimized BitLinear Layer
    Source: https://github.com/Beomi/BitNet-Transformers/tree/main
    Source License: Apache Version 2.0
    """

    def __init__(self, in_features, out_features, config, num_groups=1):
        super(BitLinearOptimized, self).__init__(in_features, out_features, config.bias)
        self.num_groups = num_groups
        self.eps = 1e-5

        # Initialize 1-bit quantized weights and store them as int8
        self.register_buffer(
            "quantized_weights", torch.sign(self.weight.data).to(torch.int8)
        )
        # Clear the original weights to save memory
        del self.weight

    @property
    def weight(self):
        # Return the dequantized weights when accessed
        return self.dequantize_weights()

    @weight.setter
    def weight(self, value):
        # Update the quantized_weights when the weight property is set
        self.quantized_weights.data = torch.sign(value).to(torch.int8)

    def dequantize_weights(self):
        # Convert quantized_weights back to bfloat16 and compute alpha for the weights
        bfloat16_weights = self.quantized_weights.to(torch.bfloat16)
        alpha = bfloat16_weights.mean()
        return bfloat16_weights * alpha

    def ste_binarize(self, x):
        # Apply the sign function for binarization
        binarized_x = torch.sign(x)
        # Use STE: during backward pass, we bypass the binarization
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def binarize_weights_groupwise(self):
        # Dequantize the weights before binarization
        weights = self.dequantize_weights()

        # Divide weights into groups
        group_size = weights.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(weights)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = weights[start_idx:end_idx]

            # Binarize each group using STE
            alpha_g = weight_group.mean()
            binarized_weights[start_idx:end_idx] = self.ste_binarize(
                weight_group - alpha_g
            )

        return binarized_weights

    def quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, input):
        # Binarize weights (group-wise) using STE
        binarized_weights = self.binarize_weights_groupwise()

        # Normal linear transformation with binarized weights
        output = torch.nn.functional.linear(input, binarized_weights, self.bias)

        # Quantize activations group-wise
        output = self.quantize_activations_groupwise(output)

        return output
    
class QuantizedLinear(nn.Linear):
    """Linear layer with quantization aware training capability
    Source: https://github.com/Alexstrasza98/Transformer-Quantization/blob/main
    Source License: MIT
    """

    def __init__(self, in_features, out_features, config, warmup_step=0):
        super().__init__(in_features, out_features, config.bias)

        self.config = config
        self.weight_bits = config.quantization_linear_bits

        if self.weight_bits < 1:
            raise ValueError(f"weight_bits={self.weight_bits} must be higher than 0 ")
        
        self.warmup_step = warmup_step
        self.accumulation_bits = 32

        # Placeholder for quantized weights during training
        self._fake_quantized_weight = None
        if config.bias == True:
            self.register_buffer("quantized_bias", None)
            self.register_buffer("bias_norm", None)
            self.register_buffer("bias_zero_point", torch.tensor([0]))

        self.register_buffer("_step", torch.zeros(1))

        self.register_buffer("quantized_weight", None)
        self.register_buffer("weight_norm", None)
        self.register_buffer("weight_zero_point", torch.tensor([0]))

    def training_quantized_forward(self, input):
        """Fake quantizes weights. Function should only be used while training"""
        assert self.training, "Should be called only during training"

        # Applies the fake quantization to the weights
        self._fake_quantized_weight = _fake_quantize(self.weight, self.weight_bits, self.config.quantization_linear_method)
        # Uses the quantized weights to compute the output using F.linear
        out = F.linear(input, self._fake_quantized_weight, self.bias)

        return out

    def inference_quantized_forward(self, input):
        """Simulate quantized inference. Function should be called only during inference"""
        assert not self.training, "Should be called only during inference"

        # Compute the dequantized weight
        weight = (self.quantized_weight - self.weight_zero_point[0]) * self.weight_norm

        # Compute the dequantized bias
        if self.bias is not None:
            bias = (self.quantized_bias - self.bias_zero_point[0]) * self.bias_norm

        # Uses the dequantized weights and bias to compute the output using F.linear
        if self.bias:
            out = F.linear(input, weight, bias)
        else:
            out = F.linear(input, weight)

        return out

    def _eval(self):
        """Sets the model for inference by quantizing the model"""
        self.weight_zero_point[0], self.weight_norm, self.quantized_weight = quantize_dictionary[self.config.quantization_linear_method](self.weight, self.weight_bits)

        if self.bias is not None:
            self.bias_zero_point[0], self.bias_norm, self.quantized_bias = quantize_dictionary[self.config.quantization_linear_method](self.bias, self.accumulation_bits)

    def forward(self, input):
        """Passes the input through the model during training and inference"""
        if self.training:
            if self._step > self.warmup_step:
                out = self.training_quantized_forward(input)
            else:
                out = super().forward(input)
            self._step += 1
        else:
            # Prepares the model for inference by quantizing weights and bias
            self._eval()
            # Uses quantized weights and bias to compute the output
            out = self.inference_quantized_forward(input)
        return out


class BinaryLinearFunction():
    
    """
    Implements binarization function for linear layer with Straight-Through Estimation (STE)
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # binarize weights by sign function
        weight_mask = (weight > 1) | (weight < -1)
        weight = torch.sign(weight)
        # save for grad computing
        ctx.save_for_backward(input, weight, weight_mask, bias)
        
        # linear layer
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve saved variables
        input, weight, weight_mask, bias = ctx.saved_variables
        
        # computing grads
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[1]:
            # if weights' absolute value larger than 1, no grads
            grad_weight = grad_output.transpose(-1, -2).matmul(input)
            grad_weight.masked_fill_(weight_mask, 0.0)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class BinarizedLinear(nn.Module):
    """
    Implements Binarization Layer using Binarization function
    Source: https://github.com/Alexstrasza98/Transformer-Quantization/blob/main
    Source License: MIT
    """

    def __init__(self, in_features, out_features, bias=True):
        super(BinarizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.binarized_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("b_weight", torch.Tensor(out_features, in_features))

        if bias:
            self.binarization_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.binarized_weight.data.normal_(0, 1 * (math.sqrt(1.0 / self.in_features)))
        if self.binarization_bias is not None:
            self.binarization_bias.data.zero_()

    def forward(self, input):
        if self.binarization_bias is not None:
            self.b_weight = torch.sign(self.binarized_weight)
            return BinaryLinearFunction.apply(input, self.binarized_weight, self.binarization_bias)
        else:
            raise Exception

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"

    
class IRLinearFunction():
    """
    Implements binarization function for linear layer with Straight-Through Estimation (STE)
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None, t=None):
        # normalize weights
        weight_mean = torch.mean(weight)
        weight_std = torch.std(weight)
        weight_norm = (weight - weight_mean)/weight_std
        
        # compute control variable k
        k = torch.max(torch.Tensor([1/t,1]))
        
        # binarize by EDE function
        weight_b = k * torch.tanh(t * weight_norm)
        
        # save for grad computing
        ctx.save_for_backward(input, weight_b, weight_norm, bias, weight_std, t, k)
        
        # linear layer
        output = input.matmul(weight_b.t())
        if bias is not None:
            output += bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve saved variables
        input, weight_b, weight_norm, bias, weight_std, t, k = ctx.saved_variables
        
        # computing grads
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight_b)

        if ctx.needs_input_grad[1]:
            grad_weight_b = grad_output.transpose(-1, -2).matmul(input)
            grad_binary = k * t * (1 - torch.square(torch.tanh(t * weight_norm)))
            grad_weight = grad_weight_b * grad_binary 

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None
    

class IRLinear(nn.Module):
    """
    Implements Binarization Layer using Binarization function
    Source: https://github.com/Alexstrasza98/Transformer-Quantization/blob/main
    Source License: MIT
    """

    def __init__(self, in_features, out_features, bias=True):
        super(IRLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.t = None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.t = None
        self.weight.data.normal_(0, 1 * (math.sqrt(1.0 / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.bias is not None:
            return IRLinearFunction.apply(input, self.weight, self.bias, self.t)
        else:
            raise Exception

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class TernaryLinearFunction():

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # ternarize weights
        threshold = 0.5 * torch.mean(torch.abs(weight))
        weight_mask = torch.abs(weight) < threshold
        ternary_weight = torch.sign(weight) * (~weight_mask).float()
        
        # save for grad computing
        ctx.save_for_backward(input, ternary_weight, weight_mask, bias)
        
        # linear layer
        output = input.matmul(ternary_weight.t())
        if bias is not None:
            output += bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve saved variables
        input, ternary_weight, weight_mask, bias = ctx.saved_variables
        
        # computing grads
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(ternary_weight)

        if ctx.needs_input_grad[1]:
            # if weights' absolute value less than threshold, no grads
            grad_weight = grad_output.transpose(-1, -2).matmul(input)
            grad_weight.masked_fill_(weight_mask, 0.0)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class TernarizedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(TernarizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ternarized_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("t_weight", torch.Tensor(out_features, in_features))

        if bias:
            self.ternarization_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.ternarized_weight.data.normal_(0, 1 * (math.sqrt(1.0 / self.in_features)))
        if self.ternarization_bias is not None:
            self.ternarization_bias.data.zero_()

    def forward(self, input):
        if self.ternarization_bias is not None:
            self.t_weight = self.ternarize_weights(self.ternarized_weight)
            return TernaryLinearFunction.apply(input, self.ternarized_weight, self.ternarization_bias)
        else:
            raise Exception

    def ternarize_weights(self, weight):
        threshold = 0.5 * torch.mean(torch.abs(weight))
        weight_mask = torch.abs(weight) < threshold
        ternary_weight = torch.sign(weight) * (~weight_mask).float()
        return ternary_weight

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"

linear_dictionary = {
    "linear": nn.Linear,
    "bitlinear": BitLinear,
    "bitlinear_optimized": BitLinearOptimized,
    "bitlinear_1p58": BitLinear1p58,
    "quantized_linear": QuantizedLinear,
    "binarized_linear": BinarizedLinear,
    "ternarized_linear": TernarizedLinear,
    "irlinear": IRLinear,
}
