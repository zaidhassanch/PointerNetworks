import torch
import torch.nn as nn
from transfomerz import TransformerZ
# from torch.nn.

num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4

# num_heads = 8
# num_encoder_layers = 3
# num_decoder_layers = 3
# dropout = 0.10
# max_len = 100
# forward_expansion = 4
class Transformer(nn.Module):
    def __init__(
        self,
        device,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads = 8,
        num_encoder_layers = 3,
        num_decoder_layers = 3,
        forward_expansion = 4,
        dropout = 0.10,
        max_len = 100,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)  # 7854 x 512
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)     #  100 x 512
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)  # 5893 x 512
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)     #  100 x 512

        self.device = device

        self.transformer = TransformerZ(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

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

    def forward(self, src, trg, train = 0):
        src_seq_length, N = src.shape  #::: 17x 1 >> 17x32
        trg_seq_length, N = trg.shape  #::: 1,2,..9,.. >> 21x32
        #::: src (17x32)
        #::: trg (1x1, 2x1, ... 9x1, ...).. till end of sentence >> 21x32

        #::: src_positions ([[0,0,..], [1,1,..], [2,2...], ..., [15,15,..], [16,16,...] )
        src_positions = self.computePosArray(src_seq_length, N)
        #::: trg_positions ([[0,0,..], [1,1,..], [2,2,.], ..., [7,7...], [8,8,8,])
        trg_positions = self.computePosArray(trg_seq_length, N)

        #::: src (17x1), src_embed_word (17x1x512), src_embed_pos (17x1x512), embed_src (17x1x512)
        src_embed_word = self.src_word_embedding(src)
        src_embed_pos = self.src_position_embedding(src_positions)
        embed_src = self.dropout(src_embed_word + 0*src_embed_pos)

        #::: trg (9x1x512), trg_word_embedding (9x1x512), trg_positions (9x1x512), embed_trg (9x1x512)
        trg_word_embedding = self.trg_word_embedding(trg)
        trg_positions = self.trg_position_embedding(trg_positions)
        embed_trg = self.dropout(trg_word_embedding + trg_positions)

        #::: src_padding_mask (1x17) [True, False, False,..... False]
        src_padding_mask = self.make_src_mask(src)
        # print(src_padding_mask)
        #::: src_positions (9x9) [upper triangular matrix of -inf]
        # if(train == 1):
        #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> mask length : ", trg_seq_length)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        #::: out1 (9x1x512) = embed_src(17x1x512), embed_trg(9x1x512)
        out1 = self.transformer(embed_src, embed_trg, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask)
        # out1 = self.transformer(embed_src, embed_trg)

        #::: out2 (9x1x5893)
        out2 = self.fc_out(out1)
        return out2
