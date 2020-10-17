import torch.optim as optim
from main2 import batch
import config
import time
from pointerNetwork import PointerNetwork
import torch
import torch.nn as nn

BATCH_SIZE = 32
EPOCHS = 10
STEPS_PER_EPOCH = 100

def train(pNet, optimizer, epoch, clip=1.):
  """Train single epoch"""
  print('Epoch [{}] -- Train'.format(epoch))


  for step in range(STEPS_PER_EPOCH):
    optimizer.zero_grad()
    x, y, t = batch(BATCH_SIZE)
    # Forward
    out, loss = pNet(x, y)
    # Backward
    loss.backward()
    nn.utils.clip_grad_norm_(pNet.parameters(), clip)
    optimizer.step()
    if (step + 1) % 10 == 0:
      print('Epoch [{}] loss: {}'.format(epoch, loss.item()))

def evaluateWordSort(model, epoch):
  """Evaluate after a train epoch"""
  print('Epoch [{}] -- Evaluate'.format(epoch))

  x_val, y_val, text_val = batch(4)
  out, _ = model(x_val, y_val, teacher_force_ratio=0.)
  out = out.permute(1, 0)

  for i in range(out.size(0)):
    print("=============================================")
    print("yref", y_val[i], out[i], y_val[i] - out[i])

    print("orig", text_val[i])
    v = torch.Tensor.cpu(out[i]).numpy()
    print("[", end="")
    for index in v:
      print(text_val[i][index]+" ", end="")

    print("]")

if config.GPU == True:
  ptrNet = PointerNetwork(config.HIDDEN_SIZE).cuda()
else:
  ptrNet = PointerNetwork(config.HIDDEN_SIZE)
optimizer = optim.Adam(ptrNet.parameters())

program_starts = time.time()
for epoch in range(EPOCHS):
  train(ptrNet, optimizer, epoch + 1)
  evaluateWordSort(ptrNet, epoch + 1)

now = time.time()
print("It has been {0} seconds since the loop started".format(now - program_starts))
