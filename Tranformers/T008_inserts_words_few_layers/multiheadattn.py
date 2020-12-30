import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import _LinearWithBias

def linear(input, weight, bias):
    output = input.matmul(weight.t()) + bias
    return output

class MultiheadAttentionZ(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionZ, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):

        num_heads = self.num_heads; in_proj_weight=self.in_proj_weight; in_proj_bias= self.in_proj_bias;
        out_proj_weight = self.out_proj.weight; out_proj_bias = self.out_proj.bias
        tgt_len, bsz, embed_dim = query.size()
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        _b = in_proj_bias
        if _b is not None:
            _b = _b[:embed_dim]
        q = linear(query, in_proj_weight[:embed_dim, :], _b)
        k, v = linear(key, in_proj_weight[embed_dim:, :], in_proj_bias[embed_dim:]).chunk(2, dim=-1)

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

        attn_output_weights = attn_output_weights.softmax(-1)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads

