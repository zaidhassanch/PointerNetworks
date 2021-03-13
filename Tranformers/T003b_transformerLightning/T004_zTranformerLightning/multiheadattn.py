import torch
import torch.nn as nn

class MultiheadAttentionZ(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout = 0):
        super(MultiheadAttentionZ, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, value, key_padding_mask=None, attn_mask=None):
        num_heads = self.num_heads;
        tgt_len, bsz, embed_dim = q.size()
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5

        v = self.linear(k)
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
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2),float('-inf'))
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_output_weights = attn_output_weights.softmax(-1)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        return attn_output

