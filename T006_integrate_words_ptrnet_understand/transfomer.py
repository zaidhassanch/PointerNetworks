import torch
import torch.nn as nn
import torch.optim as optim 


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
        src_pad_idx,
        num_heads = 1,
        num_encoder_layers = 3,
        num_decoder_layers = 3,
        forward_expansion = 4,
        dropout = 0.10,
        max_len = 31,
    ):
        super(Transformer, self).__init__()
        self.fc_src = nn.Linear(embedding_size, embedding_size)
        # self.fc_trg = nn.Linear(embedding_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)     #  100 x 512

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, max_len)
        self.dropout = nn.Dropout(dropout)

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
        # src_seq_length, N = src.shape  #::: 17
        # trg_seq_length, N = trg.shape  #::: 9
        src = self.fc_src(src)
        N, trg_seq_length = trg.shape
        #::: src (17x1)
        #::: trg (1x1, 2x1, ... 9x1, ...).. till end of sentence

        #::: src_positions ([[0], [1], [2], ..., [15], [16] )
        # src_positions = self.computePosArray(src_seq_length, N)
        #::: trg_positions ([[0], [1], [2], ..., [7], [8])
        # trg_positions = self.computePosArray(trg_seq_length, N)

        #::: src (17x1), src_embed_word (17x1x512), src_embed_pos (17x1x512), embed_src (17x1x512)
        # src_embed_word = self.src_word_embedding(src)
        # src_embed_pos = self.src_position_embedding(src_positions)
        # embed_src = self.dropout(src_embed_word + src_embed_pos)
        embed_src = src.permute(1, 0, 2)
        # embed_src = self.dropout(src_reshaped)

        #::: trg (9x1x512), trg_word_embedding (9x1x512), trg_positions (9x1x512), embed_trg (9x1x512)
        trg_positions = self.trg_position_embedding(trg)
        trg_positions_reshaped = trg_positions.permute(1, 0, 2)
        embed_trg = self.dropout(trg_positions_reshaped)

        #::: src_padding_mask (1x17) [True, False, False,..... False]
        #src_padding_mask = self.make_src_mask(src)
        # print(src_padding_mask)
        #::: src_positions (9x9) [upper triangular matrix of -inf]
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        #::: out1 (9x1x512) = embed_src(17x1x512), embed_trg(9x1x512)
        out1 = self.transformer(embed_src, embed_trg, tgt_mask=trg_mask)

        #::: out2 (9x1x5893)
        out2 = self.fc_out(out1)
        return out2
