"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
import sys
import re

import torch
import torch.nn as nn
from torch.nn import functional as F

# Config
from gpt_conf import GPTConfig

# Checkpointing
import torch.utils.checkpoint as checkpoint

# Variations
from variations.softmax_variations import softmax_dictionary, Softermax, ConSmax, ConSmaxQuan, SaturatingConSmax, Strongermax, Polymax, SigSoftmax, ExpPolymax, Softplus, Squareplus
from variations.norm_variations import norm_dictionary, LayerNorm, RMSNorm, pRMSNorm, kRMSNorm
from variations.position_encoding_variations import QuantizedEmbedding, RotaryEmbedding, ShortRope, SymmetricalOverlapAngularPositions, FIRE
from variations.activation_variations import SquaredReLU, activation_dictionary
from variations.linear_variations import BitLinear1p58, BitLinear, BitLinearOptimized, linear_dictionary
from quantization.quantize import quantize_dictionary, dequantize, _fake_quantize

def create_shared_param_group(layer_type, config):
    shared_size = None
    shared_sym = None # if true, output array is symmetrical
    layer_block = None
    shared_group = []

    if layer_type == "mlp":
        shared_size = config.shared_mlp_size
        shared_sym = config.shared_mlp_sym
    elif layer_type == "attn":
        shared_size = config.shared_attn_size
        shared_sym = config.shared_attn_sym
    else:
        sys.exit(f"{layer_type} not supported, exiting")

    # if attn layer check if using shared fire embeddings
    fire_pos_enc = None
    if layer_type == "attn" and config.shared_fire_embeddings:
        fire_pos_enc = FIRE(num_heads=config.n_head)

    for i in range (config.n_layer):

        # Create new layer block every "shared_size"
        if i % shared_size == 0:
            if layer_type == "mlp":
                layer_block = MLP(config)
            elif layer_type == "attn":
                layer_block = CausalSelfAttention(config, fire_pos_enc=fire_pos_enc)
            else:
                sys.exit(f"{layer_type} not supported, exiting")

        # Add layer block
        shared_group.append(layer_block)

        # If symmetrical and halfway, then mirror extend and exit
        if shared_sym:
            # Even
            if config.n_layer % 2 == 0:
                if i == (config.n_layer // 2 - 1):
                    # Append going backwards
                    for j in range(i+1):
                        shared_group.append(shared_group[i - j])
                    return shared_group
            # Odd
            else:
                if i == (config.n_layer // 2):
                    # Append going backwards
                    for j in range(i):
                        shared_group.append(shared_group[i - j])
                    return shared_group
    return shared_group

def quantize_helper(training, tensor, num_bits, quantization_linear_method):
    if training:
        return _fake_quantize(tensor, num_bits)
    else:
        zero_point, quantized_norm, quantized_matrix = quantize_dictionary[quantization_linear_method](tensor, num_bits)
        return dequantize(zero_point, quantized_norm, quantized_matrix)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, fire_pos_enc=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.linear_variant = linear_dictionary[config.linear_variant]
        # key, query, value projections for all heads, but in a batch
        if config.linear_variant != "linear" and (config.quantize_c_attn_q or config.quantize_attn_all):
            self.c_attn_q = self.linear_variant(config.n_embd, config.n_embd, config=config)
        else:
            self.c_attn_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.n_head = config.n_head
        if config.n_kv_group == None:
            self.n_kv_group = config.n_head
        else:
            assert config.n_head % config.n_kv_group == 0
            self.n_kv_group = config.n_kv_group

        self.kv_dim = (config.n_embd // config.n_head) * self.n_kv_group

        self.c_attn_k = nn.Linear(config.n_embd, self.kv_dim, bias=config.bias)
        self.c_attn_v = nn.Linear(config.n_embd, self.kv_dim, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        if config.linear_variant != "linear" and (config.quantize_c_attn_k or config.quantize_attn_all):
            self.c_attn_k = self.linear_variant(config.n_embd, self.kv_dim, config=config)
        if config.linear_variant != "linear" and (config.quantize_c_attn_v or config.quantize_attn_all):
            self.c_attn_v = self.linear_variant(config.n_embd, self.kv_dim, config=config)
        if config.linear_variant != "linear" and (config.quantize_attn_proj or config.quantize_attn_all):
            self.c_proj = self.linear_variant(config.n_embd, config.n_embd, config=config)

        # quantization
        self.quantization_bits = config.quantization_bits
        self.quantize_attn_all = config.quantize_attn_all
        self.quantization_linear_method = config.quantization_linear_method
        self.quantize_q = config.quantize_q
        self.quantize_k = config.quantize_k
        self.quantize_v = config.quantize_v
        self.quantize_q_k_mult = config.quantize_q_k_mult
        self.quantize_softmax_v_mult = config.quantize_softmax_v_mult
        self.quantize_softmax = config.quantize_softmax

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.window_size = config.window_size
        self.n_embd = config.n_embd
        self.gate = config.gate
        self.use_fire_embeddings = None
        if config.use_fire_embeddings:
            self.use_fire_embeddings = config.use_fire_embeddings
            if fire_pos_enc is not None:
                self.fire_pos_enc = fire_pos_enc
                print("shared fire")
            else:
                self.fire_pos_enc = FIRE(num_heads=config.n_head)
                print("indiv fire")

        # Rotary Positional Embeddings
        self.rotary_emb_q = None
        self.rotary_emb_k = None
        if config.use_rotary_embeddings:
            # TODO update variant name after completing rope and shortrope updates
            # TODO Add shortrope to symmetrical rope
            if config.rope_variant == "rope":
                self.sym_rot_num_angles = config.sym_rot_num_angles
                self.rotary_emb_q = SymmetricalOverlapAngularPositions(config, size=config.n_embd, num_angles=self.sym_rot_num_angles)
                self.rotary_emb_k = SymmetricalOverlapAngularPositions(config, size=self.kv_dim, num_angles=self.sym_rot_num_angles)
            # TODO update rope and shortrope to accomodate new GQA additions
            # if config.rope_variant == "rope":
            #     self.rotary_emb_q = RotaryEmbedding(config, size=config.n_embd)
            #     self.rotary_emb_k = RotaryEmbedding(config, size=config.n_embd // config.n_head * config.n_kv_group)
            # if config.rope_variant == "shortrope":
            #     self.rotary_emb_q = RotaryEmbedding(config, size=config.n_embd)
            #     self.rotary_emb_k = RotaryEmbedding(config, size=config.n_embd // config.n_head * config.n_kv_group)

        # Softmax Variant Selection
        self.softmax_variant_attn = config.softmax_variant_attn
        if self.softmax_variant_attn == "softmax":
            # Enable flash attention, which is compatible with 'softmax'
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        else:
            # Remove flash attention (only compatible with 'softmax')
            self.flash = False
            # Set softmax_layer_attn to custom softmax alternative
            self.softmax_layer_attn = softmax_dictionary[config.softmax_variant_attn](config)

        if self.window_size is not None:
            # TODO: look into supporting sliding window attn for flash attn
            self.flash = False

        if self.n_kv_group != self.n_head:
            self.flash = False

        if self.use_fire_embeddings:
            self.flash = False

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q = self.c_attn_q(x)
        k = self.c_attn_k(x)
        v = self.c_attn_v(x)

        if self.rotary_emb_q is not None:
            q = self.rotary_emb_q(q)
            k = self.rotary_emb_k(k)

        if self.window_size is not None:
            window_mask = torch.ones((1, 1, T, T), device=x.device)
            window_mask = torch.triu(window_mask, diagonal=-self.window_size)
            window_mask = self.bias[:,:,:T,:T] * window_mask

        if self.gate:
            if self.n_kv_group == self.n_head:
                Gating = nn.Linear(self.n_embd, self.n_embd, bias=True, device=x.device)
                gate_ = torch.sigmoid(Gating(x))
                q = q * gate_
                k = k * gate_
                v = v * gate_
            else:
                # TODO: Test more methods to merge Attention Gates with GQA
                # TODO: Evaluate each method's ability to even out parameter sizes
                Gating_q = nn.Linear(self.n_embd, self.n_embd, bias=True, device=x.device)
                Gating_kv = nn.Linear(self.n_embd, self.kv_dim, bias=True, device=x.device)
                gate_qx = Gating_q(x)
                gate_q = torch.sigmoid(gate_qx)
                gate_kv = torch.sigmoid(Gating_kv(gate_qx))
                q = q * gate_q
                k = k * gate_kv
                v = v * gate_kv

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_h, T, hs)
        k = k.view(B, T, self.n_kv_group, C // self.n_head).transpose(1, 2) # (B, n_kv, T, hs)
        v = v.view(B, T, self.n_kv_group, C // self.n_head).transpose(1, 2) # (B, n_kv, T, hs)

        if self.quantize_q or self.quantize_attn_all:
            q = quantize_helper(self.training, q, self.quantization_bits, self.quantization_linear_method)
        if self.quantize_k or self.quantize_attn_all:
            k = quantize_helper(self.training, k, self.quantization_bits, self.quantization_linear_method)

        y = None
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = None
            # manual implementation of attention
            if self.n_head != self.n_kv_group:
              k_repeated = k.repeat_interleave(self.n_head // self.n_kv_group, dim=1)
              att = (q @ k_repeated.transpose(-2, -1)) / math.sqrt(k.size(-1))
            else:
              att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            if self.quantize_q_k_mult or self.quantize_attn_all:
                att = quantize_helper(self.training, att, self.quantization_bits, self.quantization_linear_method)

            # apply masks
            if self.window_size is not None:
                # add mask for sliding window attention
                att = att.masked_fill(window_mask == 0, float('-inf'))
            else:
                # regular lower triangle attention
                att = att.masked_fill(self.bias[:,:,:T,:T].to(x.device) == 0, float('-inf'))

            # fire position embeddings
            if self.use_fire_embeddings is not None:
                # add learned fire bias
                att = att + self.fire_pos_enc(x)

            # softmax variation
            if self.softmax_variant_attn != 'softmax':
                if self.quantize_softmax or self.quantize_attn_all:
                    zero_point, quantized_norm, att = quantize_helper(self.training, att, self.quantization_bits, self.quantization_linear_method)
                att = self.softmax_layer_attn(att)
            else:
                if self.quantize_softmax or self.quantize_attn_all:
                    zero_point, quantized_norm, att = quantize_helper(self.training, att, self.quantization_bits, self.quantization_linear_method)
                att = F.softmax(att, dim=-1)
            
            att = self.attn_dropout(att)

            # quantize att and v
            if self.quantize_softmax or self.quantize_attn_all:
                att = (att - zero_point) * quantized_norm
            if self.quantize_v or self.quantize_attn_all:
                v = quantize_helper(self.training, v, self.quantization_bits, self.quantization_linear_method)

            if self.n_head != self.n_kv_group:
                v_repeated = v.repeat_interleave(self.n_head // self.n_kv_group, dim=1)
                y = att @ v_repeated # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            else:
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        if self.quantize_softmax_v_mult or self.quantize_attn_all:
            y = quantize_helper(self.training, y, self.quantization_bits, self.quantization_linear_method)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Select linear variant
        self.linear_variant = linear_dictionary[config.linear_variant]

        # Select activation variant
        self.activation_variant = activation_dictionary[config.activation_variant]

        # Whether the activation is quantized
        self.quantize_activation = config.quantize_activation
        self.quantization_bits = config.quantization_bits
        self.quantize_mlp_all = config.quantize_mlp_all
        self.quantization_activation_method = config.quantization_activation_method

        # Whether to ues swiglu
        self.use_swiglu = config.use_swiglu

        if self.use_swiglu:
            self.c_fc_in1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_fc_in2 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_fc_out = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            if config.linear_variant != "linear" and (config.quantize_mlp_up or config.quantize_mlp_all):
                self.c_fc_in1 = self.linear_variant(config.n_embd, 4 * config.n_embd, config=config)
                self.c_fc_in2 = self.linear_variant(config.n_embd, 4 * config.n_embd, config=config)
            if config.linear_variant != "linear" and (config.quantize_mlp_down or config.quantize_mlp_all):
                self.c_fc_out = self.linear_variant(4 * config.n_embd, config.n_embd, config=config)
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            if config.linear_variant != "linear" and (config.quantize_mlp_up or config.quantize_mlp_all):
                self.c_fc = self.linear_variant(config.n_embd, 4 * config.n_embd, config=config)
            if config.linear_variant != "linear" and (config.quantize_mlp_down or config.quantize_mlp_all):
                self.c_proj = self.linear_variant(4 * config.n_embd, config.n_embd, config=config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.use_swiglu:
            x_in1 = self.c_fc_in1(x)
            x_in1 = self.activation_variant(x_in1)
            if self.quantize_activation or self.quantize_mlp_all:
                x_in1 = quantize_helper(self.training, x_in1, self.quantization_bits, self.quantization_activation_method)
            x_in2 = self.c_fc_in2(x)
            x_out = x_in1 * x_in2
            x = self.c_fc_out(x_out)
        else:
            x = self.c_fc(x)
            x = self.activation_variant(x)
            if self.quantize_activation or self.quantize_mlp_all:
                x = quantize_helper(self.training, x, self.quantization_bits, self.quantization_activation_method)
            x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, mlp=None, attn=None):
        super().__init__()

        # Initialize and set attn normalization (e.g. rmsnorm)
        norm_variant_attn = norm_dictionary[config.norm_variant_attn]
        self.ln_1 = norm_variant_attn(config)
        if not config.use_parallel_mlp:
            self.ln_2 = norm_variant_attn(config)

        self.use_post_ln = config.use_post_ln
        self.use_parallel_mlp = config.use_parallel_mlp
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        # Allow for sharing attn between blocks
        if attn is None:
            self.attn = CausalSelfAttention(config)
        else:
            self.attn = attn

        # Allow for sharing mlp between blocks
        if mlp is None:
            self.mlp = MLP(config)
        else:
            self.mlp = mlp

    def forward(self, x):
        def custom_forward(*inputs):
            x = inputs[0]
            if self.use_post_ln:
                if self.use_parallel_mlp:
                    x = self.ln_1(x + self.attn(x) + self.mlp(x))
                else:
                    x = self.ln_1(x + self.attn(x))
                    x = self.ln_2(x + self.mlp(x))
            else:
                if self.use_parallel_mlp:
                    ln_1 = self.ln_1(x)
                    x = x + self.attn(ln_1) + self.mlp(ln_1)
                else:
                    x = x + self.attn(self.ln_1(x))
                    x = x + self.mlp(self.ln_2(x))
            return x

        if self.use_gradient_checkpointing and x.requires_grad:
            return checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
        else:
            return custom_forward(x)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        # Initialize and set ouptut normalization (e.g. rmsnorm)
        self.norm_variant_output = norm_dictionary[config.norm_variant_output](config)

        # Shared Parameters MLP
        shared_mlp_array = create_shared_param_group("mlp", config)
        # Shared Parameters Attention
        shared_attn_array = create_shared_param_group("attn", config)

        if config.quantize_wte:
            word_embd = QuantizedEmbedding(config.vocab_size, config.n_embd, config)
        else:
            word_embd = nn.Embedding(config.vocab_size, config.n_embd)
        if config.quantize_wpe:
            pos_embd = QuantizedEmbedding(config.block_size, config.n_embd, config)
        else:
            pos_embd = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte = word_embd,
            wpe = pos_embd,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, mlp=shared_mlp_array[i], attn=shared_attn_array[i]) for i in range(config.n_layer)]),
            ln_f = self.norm_variant_output,
        ))

        # Select softmax variant for output layer
        self.softmax_variant_output = config.softmax_variant_output
        if self.softmax_variant_output != "softmax":
            self.softmax_layer_output = softmax_dictionary[config.softmax_variant_output](config)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def update_block_size(self, new_block_size):
        # Function to increase block size dynamically
        if new_block_size > self.config.block_size:
            self.config.block_size = new_block_size
            self.transformer.wpe = nn.Embedding(new_block_size, self.config.n_embd)
            for block in self.transformer.h:
                if hasattr(block.attn, 'bias'):
                    block.attn.bias = torch.tril(torch.ones(new_block_size, new_block_size)).view(1, 1, new_block_size, new_block_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def update_num_angles(self, num_angles):
        """Update the number of angles for rotary embeddings in all attention layers."""
        device = next(self.parameters()).device
        for block in self.transformer.h:
            if hasattr(block.attn, 'rotary_emb_q') and hasattr(block.attn, 'rotary_emb_k'):
                block.attn.rotary_emb_q.update_num_angles(num_angles, device)
                block.attn.rotary_emb_k.update_num_angles(num_angles, device)


    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = None
        if self.config.use_abs_pos_embeddings:
          pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
          pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
          x = self.transformer.drop(tok_emb + pos_emb)
        else:
          x = self.transformer.drop(tok_emb)

        x.requires_grad_(True)  # Ensure requires_grad is True

        for block in self.transformer.h:
            if self.config.use_gradient_checkpointing:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        config_args['window_size'] = 128 # always None for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = None
            if self.config.softmax_variant_output != 'softmax':
                probs = self.softmax_layer_output(logits)
            else:
                probs = F.softmax(logits, dim=-1)
            assert probs != None
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_with_stop(self, idx, max_new_tokens, stop_string, decode, temperature=1.0, top_k=None):
        """
        Generate tokens and stop on fixed string match, return the state for further input.
        """
        generated_text = ""
        buffer = ""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            next_token_text = decode(idx_next[0].tolist())
            generated_text += next_token_text
            buffer += next_token_text

            # Check if the buffer ends with the stop_string
            if buffer.endswith(stop_string):
                break

        return idx, generated_text
