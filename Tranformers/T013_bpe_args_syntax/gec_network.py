import torch
import torch.nn as nn
from rswi_transformer import RSWITransformer

class GECNetwork(nn.Module):
    def __init__(
            self,
            device,
            embedding_size,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            arch_flag="END",
            syntax_embedding_size=256,
            num_heads=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            forward_expansion=4,
            dropout=0.10,
            max_len=100):
        super(GECNetwork, self).__init__()
        #embedding_size = 256
        self.rswi_trans = RSWITransformer(device, embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx,
                                arch_flag=arch_flag, syntax_embedding_size=syntax_embedding_size).to(device)

    def forward(self, src, trg, train=0, syntax_embedding=False):
        #print("forward")
        out = self.rswi_trans(src, trg, train, syntax_embedding)
        return out
