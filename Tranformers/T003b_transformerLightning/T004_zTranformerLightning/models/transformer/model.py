import time
import torch
import torch.nn as nn
from models.transformer.transformerz import TransformerZ
from configs import config



# dropout = 0.10
# max_len = 100
# forward_expansion = 4

# num_heads = 8
# num_encoder_layers = 3
# num_decoder_layers = 3
# dropout = 0.10
# max_len = 100
# forward_expansion = 4
class Model(nn.Module):
    def __init__(
        self,
        device,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads = config.NUM_HEADS,
        num_encoder_layers = config.N_LAYERS,
        num_decoder_layers = config.N_LAYERS,
        forward_expansion = config.FORWARD_EXP,
        dropout = 0.0,
        max_len = config.MAX_LEN,
    ):
        super(Model, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)  # 7854 x 512
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)     #  100 x 512
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)  # 5893 x 512
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)     #  100 x 512

        self.device = device
        flag = config.PYTORCH_TRANSFORMER
        if flag:
            self.transformer = nn.Transformer(
                embedding_size,
                num_heads,
                num_encoder_layers,
                num_decoder_layers,
                forward_expansion,
                dropout,
            )

        else:
            self.transformer = TransformerZ(
                embedding_size,
                num_heads, #num_heads,
                num_encoder_layers,
                num_decoder_layers,
                forward_expansion,
                dropout,
            )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        self.trg_vocab_size = trg_vocab_size

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def computePosArray(self, seqLen, N):
        positions = (
            torch.arange(0, seqLen)
                .unsqueeze(1)
                .expand(seqLen, N)
                .to(self.device)
        )
        return positions

    def forward(self, src, trg):
        src_seq_length, N = src.shape  #::: 17x 1 >> 17x32
        trg_seq_length, N = trg.shape  #::: 1,2,..9,.. >> 21x32

        src_positions = self.computePosArray(src_seq_length, N)
        trg_positions = self.computePosArray(trg_seq_length, N)

        src_embed_word = self.src_word_embedding(src)
        src_embed_pos = self.src_position_embedding(src_positions)
        embed_src = self.dropout(src_embed_word + src_embed_pos)

        trg_word_embedding = self.trg_word_embedding(trg)
        trg_positions = self.trg_position_embedding(trg_positions)
        embed_trg = self.dropout(trg_word_embedding + trg_positions)

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)
        # t = time.time()
        out1 = self.transformer(embed_src, embed_trg, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask)
        out2 = self.fc_out(out1)
        # out3 = torch.autograd.Variable(torch.zeros(trg_seq_length, N, self.trg_vocab_size).to(self.device), requires_grad = True)
        return out2
