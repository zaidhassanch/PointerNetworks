import random
from typing import Tuple
from test import computeAttn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from train import train, evaluate, BATCH_SIZE

HIDDEN_SIZE = 256
EPOCHS = 10


class Attention(nn.Module):
  def __init__(self, hidden_size, units):
    super(Attention, self).__init__()
    self.W1 = nn.Linear(hidden_size, units, bias=False)
    self.W2 = nn.Linear(hidden_size, units, bias=False)
    self.V =  nn.Linear(units, 1, bias=False)
    return

  def forward(self, 
              encoder_out: torch.Tensor, 
              decoder_hidden: torch.Tensor):
    decoder_hidden_time = decoder_hidden.unsqueeze(1)
    uj = self.W1(encoder_out) + self.W2(decoder_hidden_time)
    uj = torch.tanh(uj)
    uj = self.V(uj)
    aj = F.softmax(uj, dim=1)
    di_prime = aj * encoder_out
    di_prime = di_prime.sum(1)
    uj = uj.squeeze(-1)
    return di_prime, uj

class Decoder(nn.Module):
  def __init__(self, 
               hidden_size: int,
               attention_units: int = 10):
    super(Decoder, self).__init__()
    self.lstm = nn.LSTM(hidden_size + 1, hidden_size, batch_first=True)
    self.attention = Attention(hidden_size, attention_units)
    self.hs = torch.zeros(BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float)

    return

  def forward(self, 
              x: torch.Tensor, 
              hidden: Tuple[torch.Tensor], 
              encoder_out: torch.Tensor):

    di, att_w = self.attention(encoder_out, hidden)
    x = torch.cat([di.unsqueeze(1), x], dim=2)
    _, hidden = self.lstm(x)
    # self.hs =
    return hidden[0][0], att_w


class PointerNetwork(nn.Module):
  def __init__(self):
    super(PointerNetwork, self).__init__()
    self.encoder = nn.LSTM(1, HIDDEN_SIZE, batch_first=True)
    self.decoder = Decoder(HIDDEN_SIZE)

  def forward(self, 
              x: torch.Tensor, 
              y: torch.Tensor, 
              teacher_force_ratio=.5):

    encoder_in = x.unsqueeze(-1).type(torch.float)
    out, _ = self.encoder(encoder_in)
    loss = 0

    outputs = torch.zeros(out.size(1), out.size(0), dtype=torch.long)
    
    dec_in = torch.zeros(out.size(0), 1, 1, dtype=torch.float)
    hs = torch.zeros(out.size(0), out.size(2), dtype=torch.float)
    for t in range(out.size(1)):
      hs, att_w = self.decoder(dec_in, hs, out)
      predictions = F.softmax(att_w, dim=1).argmax(1)

      # Pick next index
      # If teacher force the next element will we the ground truth
      # otherwise will be the predicted value at current timestep
      teacher_force = random.random() < teacher_force_ratio
      idx = y[:, t] if teacher_force else predictions
      dec_in = torch.stack([x[b, idx[b].item()] for b in range(x.size(0))])
      dec_in = dec_in.view(out.size(0), 1, 1).type(torch.float)

      # Add cross entropy loss (F.log_softmax + nll_loss)
      loss += F.cross_entropy(att_w, y[:, t])
      outputs[t] = predictions

    # Weight losses, so every element in the batch 
    # has the same 'importance' 
    batch_loss = loss / y.size(0)

    return outputs, batch_loss


ptr_net = PointerNetwork()

optimizer = optim.Adam(ptr_net.parameters())

for epoch in range(EPOCHS):
  train(ptr_net, optimizer, epoch + 1)
  evaluate(ptr_net, epoch + 1)
  