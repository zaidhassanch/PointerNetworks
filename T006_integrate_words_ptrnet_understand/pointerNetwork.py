"""
Module implementing the pointer network proposed at: https://arxiv.org/abs/1506.03134

ei: Encoder hidden state

di: Decoder hidden state
di_prime: Attention aware decoder state
"""
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#from data import  batch
import config
import time
from test import computeAttn

BATCH_SIZE = 32


class Encoder(nn.Module):
  def __init__(self, hidden_size: int):
    super(Encoder, self).__init__()
    self.lstm = nn.LSTM(config.NUM_FEATURES, hidden_size, batch_first=True)
  
  def forward(self, x: torch.Tensor):
    # x: (BATCH, ARRAY_LEN, 1)
    return self.lstm(x)


class Attention(nn.Module):
  def __init__(self, hidden_size, units):
    super(Attention, self).__init__()
    self.W1 = nn.Linear(hidden_size, units, bias=False)
    self.W2 = nn.Linear(hidden_size, units, bias=False)
    self.V =  nn.Linear(units, 1, bias=False)

  def forward(self, 
              encoder_out: torch.Tensor, 
              decoder_hidden: torch.Tensor):
    # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
    # encoder_out: (32, 8, 256)
    # decoder_hidden: (BATCH, HIDDEN_SIZE)
    # decoder_hidden: (32, 256)

    # Add time axis to decoder hidden state
    # in order to make operations compatible with encoder_out
    # decoder_hidden_time: (BATCH, 1, HIDDEN_SIZE)
    decoder_hidden_time = decoder_hidden.unsqueeze(1)

    # uj: (BATCH, ARRAY_LEN, ATTENTION_UNITS)
    # Note: we can add the both linear outputs thanks to broadcasting
    # ve: 32 x 8 x 10
    ve = self.W1(encoder_out);
    # vd: 32 x 1 x 10
    vd = self.W2(decoder_hidden_time)

    # uj1: (32 x 8 x 10)
    uj1 = ve + vd


    # uj1: (32 x 8 x 10)
    uj2 = torch.relu(uj1)
    #::: ve (32 x 8 x 10)
    #::: vd (32 x 1 x 10)
    #::: uj2 (32 x 8 x 10)

    # uj: (BATCH, ARRAY_LEN, 1)
    # uj3: (32 x 8 x 1)
    uj3 = self.V(uj2)

    # Attention mask over inputs
    # aj: (BATCH, ARRAY_LEN, 1)
    # aj = F.softmax(uj3, dim=1)

    # di_prime: (BATCH, HIDDEN_SIZE)
    # di_prime = aj * encoder_out
    # di_prime = di_prime.sum(1)

    di_prime = computeAttn(encoder_out, decoder_hidden_time)

    # uj4: (32 x 8) out for loss comparision
    uj4 = uj3.squeeze(-1)
    return di_prime, uj4
    

class Decoder(nn.Module):
  def __init__(self, 
               hidden_size: int,
               attention_units: int = 10):
    super(Decoder, self).__init__()
    self.lstm = nn.LSTM(hidden_size + config.NUM_FEATURES, hidden_size, batch_first=True)
    # self.lstm = nn.LSTM(10, hidden_size, batch_first=True)

    self.attention = Attention(hidden_size, attention_units)

  def forward(self, 
              dec_in: torch.Tensor,
              hidden: Tuple[torch.Tensor], 
              encoder_out: torch.Tensor):
    #::: ####################
    #:::   DECODER
    #::: ####################
    # x: (BATCH, 1, 1)
    # hidden: (1, BATCH, HIDDEN_SIZE)
    # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
    # For a better understanding about hidden shapes read: https://pytorch.org/docs/stable/nn.html#lstm
    
    # Get hidden states (not cell states) 
    # from the first and unique LSTM layer 
    # hidden = 2 x 32 x 256                                                                 : ZHC
    #   the first dim is of tuple h0, c0, basically it is decoder hidden state of dim = 256 : ZHC
    ht = hidden[0][0]  # ht: (BATCH, HIDDEN_SIZE)

    #::: di: Attention aware encoder output state based on all encoder hidden states as K, V,  : ZHC
    #:::     and decoder hidden state as query. All of dimension 256                           : ZHC
    #::: att_w: Not 'softmaxed', torch will take care of it in crossEntropy -> (32 x 8)
    di, att_w = self.attention(encoder_out, ht)
    #::: di (32 x 256)
    # Append attention aware hidden state to our input
    # x: (BATCH, 1, 1 + HIDDEN_SIZE)
    #::: dec_in (32 x 1 x 6)
    #::: lstm_in (32 x 1 x 262)

    lstm_in = torch.cat([di.unsqueeze(1), dec_in], dim=2)
    
    # Generate the hidden state for next timestep
    _, hidden = self.lstm(lstm_in, hidden)
    return hidden, att_w


class PointerNetwork(nn.Module):
  def __init__(self, hiddenSize):
    super(PointerNetwork, self).__init__()
    encoder = Encoder(hiddenSize)
    decoder = Decoder(hiddenSize)
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, 
              x: torch.Tensor, 
              y: torch.Tensor, 
              teacher_force_ratio=.5):

    #:::####################
    #::: Encoder
    #:::####################

    #::: x = (32 x 8 x 6), (BATCH x SEQ_LEN x NUM_FEATURES), we have 6 lettered words, each alphabet is a feature
    #::: y = (32 x 8), (BATCH x SEQ_LEN), The desired order of the indices

    encoder_in = x.type(torch.float)  #convert integers to floats

    #::: out: (32 x 8 x 256) (BATCH, SEQ_LEN, HIDDEN_SIZE)
    out, _ = self.encoder(encoder_in)

    loss = 0

    # Save outputs at each timestep
    #::: outputs: (8 x 32)(SEQ_LEN, BATCH), EACH COLUMN GIVES US THE DESIRED SEQ, we will update one column in each time-step
    outputs = torch.zeros(out.size(1), out.size(0), dtype=torch.long)
    
    #::: dec_in: (32 x 1 x 6) (BATCH, 1, NUM_FEATURES), should be start of sentence token
    dec_in = torch.zeros(out.size(0), 1, config.NUM_FEATURES, dtype=torch.float)
    hsa = torch.zeros(1, out.size(0), out.size(2), dtype=torch.float)
    hs = (hsa, hsa)
    #::: hsa: (1 x 32 x 256) (BATCH, 1, NUM_FEATURES)

    for t in range(out.size(1)):
      hs, att_w = self.decoder(dec_in, hs, out)
      loss += F.cross_entropy(att_w, y[:, t])

      predictions = F.softmax(att_w, dim=1).argmax(1)

      teacher_force = random.random() < teacher_force_ratio
      idx = y[:, t] if teacher_force else predictions

      #::: v is (32 x 6)  (x is 32 x 8 x 6)
      v = [];
      for b in range(x.size(0)):
        v.append(x[b, idx[b].item()])

      dec_in = torch.stack(v)

      dec_in = dec_in.view(out.size(0), 1, config.NUM_FEATURES).type(torch.float)

      outputs[t] = predictions

    batch_loss = loss / y.size(0)

    return outputs, batch_loss


