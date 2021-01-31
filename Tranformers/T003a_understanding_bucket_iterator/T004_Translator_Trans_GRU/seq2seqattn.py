import torch
import random
from torch import nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, num_words, emb_dim, hid_dim):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(num_words,emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)

    def forward(self, input):
        embedded_word = self.embedding(input)
        _, hidden = self.gru(embedded_word)
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):
        # print(input.shape)
        output = self.embedding(input)
        # print(output.shape)
        output = F.relu(output)

        dec_out, hidden = self.gru(output, hidden)
        dec_out = dec_out.squeeze(0)
        output = self.fc(dec_out)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, src_pad_idx, src_vocab_size, trg_vocab_size, device):
        super().__init__()
        ENC_EMB_DIM = 300;	DEC_EMB_DIM = 300
        ENC_HID_DIM = 512;  DEC_HID_DIM = 512
        ENC_DROPOUT = 0.2;  DEC_DROPOUT = 0.2

        self.encoder = EncoderRNN(src_vocab_size, ENC_EMB_DIM, ENC_HID_DIM)
        self.decoder = DecoderRNN(trg_vocab_size, DEC_EMB_DIM, DEC_HID_DIM)
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.input_dim = src_vocab_size
        self.output_dim = trg_vocab_size

    def forward(self, src, trg, teacher_forcing_ratio = 0.50):
        trg_len = trg.shape[0]
        batch_size = src.shape[1]
        trg_vocab_size = self.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        enc_hidden = self.encoder(src)
        #enc_hidden = torch.cat((syntax, content), dim = 1)
        # enc_hidden = enc_hidden.unsqueeze(0)
        hidden = enc_hidden
        target = trg[0,:]
        #exit()

        for t in range(0, trg_len):
            output, hidden = self.decoder(target.unsqueeze(0), hidden)

            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            target = trg[t] #if teacher_force else top1
        return outputs



# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, input, hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
#
#
# class AttnDecoderRNN(nn.Module):
#
#     def __init__(self, hidden_size, num_words, dropout_p=0.1):
#         super(AttnDecoderRNN, self).__init__()
#
#         self.num_words = num_words
#         self.hidden_size = hidden_size
#
#         self.embedding = nn.Embedding(num_words, hidden_size)
#         self.dropout = nn.Dropout(p=dropout_p)
#         self.attn = nn.Linear(self.hidden_size*2, config.MAX_LENGTH)
#
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.num_words)
#
#     def forward(self, input, hidden, encoder_outputs):
#
#         embed_word = self.embedding(input).view(1, 1, -1)
#         embeded_word = self.dropout(embed_word)
#
#         attn_weights = F.softmax(
#             self.attn(torch.cat((embeded_word, hidden), 2)), dim=2
#         )
#
#         attn_applied = torch.bmm(attn_weights, encoder_outputs.unsqueeze(0))
#         attn_comb = F.relu(self.attn_combine(torch.cat((embed_word, attn_applied), 2)))
#         output , hidden = self.gru(attn_comb, hidden)
#
#         output = F.log_softmax(self.out(output[0]), dim=1)
#
#         return output, hidden, attn_weights
#
#
#     def initHidden(self):
#         return torch.zeros((1, 1, self.hidden_size), device=config.device)
#
