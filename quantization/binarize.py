import math
import torch
import torch.nn as nn
from torch.autograd import Function
import pdb

class BinaryLinearFunction(Function):
    
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

    
class IRLinearFunction(Function):
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
        
#         pdb.set_trace()
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


class TernaryLinearFunction(Function):

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