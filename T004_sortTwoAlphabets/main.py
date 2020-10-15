from ptr_net import Encoder, Decoder, PointerNetwork, train, evaluate
import time
import torch.optim as optim
import config
from data import batch

EPOCHS = 1
BATCH_SIZE = 32


ptrNet = PointerNetwork(config.HIDDEN_SIZE)

optimizer = optim.Adam(ptrNet.parameters())

program_starts = time.time()
for epoch in range(EPOCHS):

  x, y = batch(BATCH_SIZE)
  train(x, y, ptrNet, optimizer, epoch + 1)

  x_val, y_val = batch(4)
  evaluate(x_val, y_val, ptrNet, epoch + 1)

now = time.time()
print("It has been {0} seconds since the loop started".format(now - program_starts))
