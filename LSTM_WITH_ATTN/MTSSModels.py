import torch
import torch.nn as nn
import torch.nn.functional as F

import config

torch.manual_seed(0)

device = "cuda"

MAX_LENGTH = 10


class EncoderRNN(nn.Module):
    def __init__(self, num_words, hidden_size):
        super(EncoderRNN, self).__init__()
        self.num_words = num_words
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_words,hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded_word = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded_word, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros((1,1,self.hidden_size), device=config.device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, num_words, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        self.num_words = num_words
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_words, hidden_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.attn = nn.Linear(self.hidden_size*2, config.MAX_LENGTH)

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.num_words)

    def forward(self, input, hidden, encoder_outputs):

        embed_word = self.embedding(input).view(1, 1, -1)
        embeded_word = self.dropout(embed_word)

        attn_weights = F.softmax(
            self.attn(torch.cat((embeded_word, hidden), 2)), dim=2
        )

        attn_applied = torch.bmm(attn_weights, encoder_outputs.unsqueeze(0))
        attn_comb = F.relu(self.attn_combine(torch.cat((embed_word, attn_applied), 2)))
        output , hidden = self.gru(attn_comb, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights


    def initHidden(self):
        return torch.zeros((1, 1, self.hidden_size), device=config.device)




