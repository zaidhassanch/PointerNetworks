import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data import sample, batch

STEPS_PER_EPOCH = 500
BATCH_SIZE = 32

def train(model, optimizer, epoch, clip=1.):
  """Train single epoch"""
  print('Epoch [{}] -- Train'.format(epoch))
  for step in range(STEPS_PER_EPOCH):
    optimizer.zero_grad()

    # Forward
    x, y = batch(BATCH_SIZE)
    out, loss = model(x, y)

    # Backward
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    if (step + 1) % 100 == 0:
      print('Epoch [{}] loss: {}'.format(epoch, loss.item()))


@torch.no_grad()
def evaluate(model, epoch):
  """Evaluate after a train epoch"""
  print('Epoch [{}] -- Evaluate'.format(epoch))

  x_val, y_val = batch(4)
  
  out, _ = model(x_val, y_val, teacher_force_ratio=0.)
  out = out.permute(1, 0)

  for i in range(out.size(0)):
    print('{} --> {} --> {}'.format(
      x_val[i], 
      x_val[i].gather(0, out[i]), 
      x_val[i].gather(0, y_val[i])
    ))

