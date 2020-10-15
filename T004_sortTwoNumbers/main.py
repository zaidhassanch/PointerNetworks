import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import batch
import config
import time
from ptr_net import Encoder, Decoder, PointerNetwork, train, evaluate

HIDDEN_SIZE = 256

BATCH_SIZE = 32
STEPS_PER_EPOCH = 500
EPOCHS = 10


ptr_net = PointerNetwork()

optimizer = optim.Adam(ptr_net.parameters())

program_starts = time.time()
for epoch in range(EPOCHS):
  train(ptr_net, optimizer, epoch + 1)
  evaluate(ptr_net, epoch + 1)

now = time.time()
print("It has been {0} seconds since the loop started".format(now - program_starts))

  