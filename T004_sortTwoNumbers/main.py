import random
import torch.optim as optim

from data import batch
import config
import time
from ptr_net import PointerNetwork
import torch
import torch.nn as nn

BATCH_SIZE = 32
# STEPS_PER_EPOCH = 500
EPOCHS = 10
STEPS_PER_EPOCH = 500


def train(pNet, optimizer, epoch, clip=1.):
  """Train single epoch"""
  print('Epoch [{}] -- Train'.format(epoch))
  for step in range(STEPS_PER_EPOCH):
    optimizer.zero_grad()

    # Forward
    x, y = batch(BATCH_SIZE)
    out, loss = pNet(x, y)

    # Backward
    loss.backward()
    nn.utils.clip_grad_norm_(pNet.parameters(), clip)
    optimizer.step()

    if (step + 1) % 10 == 0:
      print('Epoch [{}] loss: {}'.format(epoch, loss.item()))

@torch.no_grad()
def evaluate(model, epoch):
  """Evaluate after a train epoch"""
  print('Epoch [{}] -- Evaluate'.format(epoch))

  x_val, y_val = batch(4)
  
  out, _ = model(x_val, y_val, teacher_force_ratio=0.)
  out = out.permute(1, 0)

  sumVal = x_val.sum(dim=2)
  for i in range(out.size(0)):
    print('{} --> {} --> {} --> {}'.format(
      sumVal[i], 
      sumVal[i].gather(0, out[i]),
      sumVal[i].gather(0, y_val[i]),
      sumVal[i].gather(0, y_val[i]) - sumVal[i].gather(0, out[i])
    ))


ptrNet = PointerNetwork(config.HIDDEN_SIZE)

optimizer = optim.Adam(ptrNet.parameters())

program_starts = time.time()
for epoch in range(EPOCHS):
  train(ptrNet, optimizer, epoch + 1)
  evaluate(ptrNet, epoch + 1)


now = time.time()
print("It has been {0} seconds since the loop started".format(now - program_starts))

  