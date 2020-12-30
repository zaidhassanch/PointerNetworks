import copy
from typing import Optional, Any

import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.modules import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from torch.nn.parameter import Parameter
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.init import constant_

from torch._jit_internal import boolean_dispatch, List, Optional, _overload, Tuple

def softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype is None:
        ret = input.softmax(dim)
    return ret


def linear(input, weight, bias=None):
    output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    ret = output
    return ret

def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,

                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 ):

    tgt_len, bsz, embed_dim = query.size()
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5

    if torch.equal(query, key) and torch.equal(key, value):
        q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

    elif torch.equal(key, value):

        _b = in_proj_bias
        _start = 0
        _end = embed_dim
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = linear(query, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = embed_dim
        _w = in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        k, v = linear(key, _w, _b).chunk(2, dim=-1)

    q = q * scaling
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))

    if attn_mask is not None:
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads

class MultiheadAttentionZ(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionZ, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim =  embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        self.bias_k = self.bias_v = None

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):

        return multi_head_attention_forward(
            query, key, value, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.out_proj.weight, self.out_proj.bias,

            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask)

