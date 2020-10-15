import random
import torch.optim as optim

from data import batch
import config
import time
from ptr_net import PointerNetwork, train, evaluate

HIDDEN_SIZE = 256

BATCH_SIZE = 32
STEPS_PER_EPOCH = 500
EPOCHS = 10


ptrNet = PointerNetwork()

optimizer = optim.Adam(ptrNet.parameters())

program_starts = time.time()
for epoch in range(EPOCHS):
  x, y = batch(BATCH_SIZE)
  train(ptrNet, x, y, optimizer, epoch + 1)

  evaluate(ptrNet, epoch + 1)


  # x_val, y_val = batch(4)
  # out, _ = ptrNet(x_val, y_val, teacher_force_ratio=0.)


now = time.time()
print("It has been {0} seconds since the loop started".format(now - program_starts))

  