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
        output, hidden = self.gru(embedded_word)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input, hidden, enc_out = None):
        input = input.unsqueeze(0)
        embed = self.embedding(input)
        embeded = self.relu(embed)

        dec_out, hidden = self.gru(embeded, hidden)
        dec_out = dec_out.squeeze(0)
        output = self.fc(dec_out)
        return output, hidden

class DecoderRNNattn(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout_p=0.1):
        super(DecoderRNNattn, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)

        MAX_LENGTH = 20
        self.dropout = nn.Dropout(p=dropout_p)
        self.attn = nn.Linear(emb_dim+hid_dim, MAX_LENGTH)
        self.attn_combine = nn.Linear(hid_dim * 2, emb_dim)

        self.gru = nn.GRU(emb_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input, hidden, encoder_outputs):   #encoder_outputs  12x1x512
        input = input.unsqueeze(0)
        embed_word = self.embedding(input)

        embeded_word = self.dropout(embed_word)
        concat1 = torch.cat((embeded_word, hidden), 2)    # 1x1x812
        out = self.attn(concat1)                          # 1x1x20
        attn_weights = F.softmax(out, dim=2)              # 1x1x20
        enc_outs_reshaped = encoder_outputs.permute(1,2,0)

        #(32x1x12) = (32x1x512)(32x512x12)
        attn_applied = torch.bmm(attn_weights, enc_outs_reshaped)    # (1,1,20)(12x1x512)
        # concat2 = torch.cat((embed_word, attn_applied), 2)
        # attw = self.attn_combine(concat2)
        # attn_comb = F.relu(attw)
        # dec_out, hidden = self.gru(attn_comb, hidden)

        embed_word = self.relu(embed_word)
        dec_out, hidden = self.gru(embed_word, hidden)
        dec_out = dec_out.squeeze(0)
        output = self.fc(dec_out)
        return output, hidden


class AttnDecoderRNN(nn.Module):

    def __init__(self, num_words, hidden_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        MAX_LENGTH = 20
        self.num_words = num_words
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_words, hidden_size)

        self.dropout = nn.Dropout(p=dropout_p)
        self.attn = nn.Linear(self.hidden_size*2, MAX_LENGTH)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.num_words)

    def forward(self, input, hidden, encoder_outputs):

        embed_word = self.embedding(input)

        embeded_word = self.dropout(embed_word)
        out = self.attn(torch.cat((embeded_word, hidden), 2))
        attn_weights = F.softmax(out, dim=2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs.unsqueeze(0))
        attn_comb = F.relu(self.attn_combine(torch.cat((embed_word, attn_applied), 2)))

        output , hidden = self.gru(attn_comb, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, src_pad_idx, src_vocab_size, trg_vocab_size, device):
        super().__init__()
        ENC_EMB_DIM = 300;	DEC_EMB_DIM = 300
        HID_DIM = 512

        self.encoder = EncoderRNN(src_vocab_size, ENC_EMB_DIM, HID_DIM)
        self.decoder = DecoderRNNattn(trg_vocab_size, DEC_EMB_DIM, HID_DIM)
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.input_dim = src_vocab_size
        self.output_dim = trg_vocab_size

    def forward(self, src, trg, teacher_forcing_ratio = 0.50):
        trg_len = trg.shape[0]
        batch_size = src.shape[1]
        trg_vocab_size = self.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        enc_out, hidden = self.encoder(src)
        target = trg[0,:]

        for t in range(0, trg_len):
            output, hidden = self.decoder(target, hidden, enc_out)

            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            target = trg[t] if teacher_force else top1
        return outputs




