# import torch.optim as optim
from dataGen.dataGenerator import batch, convertToWordSingle
import config
import time
from pointerNetwork import PointerNetwork
import torch

BATCH_SIZE = 32
EPOCHS = 50
STEPS_PER_EPOCH = 500

def train(pNet, optimizer, epoch, clip=1.):

  print('Epoch [{}] -- Train'.format(epoch))
  for step in range(STEPS_PER_EPOCH):
    optimizer.zero_grad()
    x, y = batch(BATCH_SIZE)
    out, loss = pNet(x, y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(pNet.parameters(), clip)
    optimizer.step()

    if (step + 1) % 100 == 0:
      print('Epoch [{}] loss: {}'.format(epoch, loss.item()))

def evaluateWordSort(model, epoch):
  """Evaluate after a train epoch"""
  print('Epoch [{}] -- Evaluate'.format(epoch))

  x_val, y_val = batch(4)
  out, _ = model(x_val, y_val, teacher_force_ratio=0.)
  out = out.permute(1, 0)

  for i in range(out.size(0)):
    print("=============================================")
    print("yref", y_val[i], out[i], y_val[i] - out[i], )

    xv = convertToWordSingle(x_val[i])
    print("orig", xv)
    v = out[i].numpy()
    print("[", end="")
    for index in v:
      print(xv[index] + ", ", end="")

    print("]")

ptrNet = PointerNetwork(config.HIDDEN_SIZE)
optimizer = torch.optim.Adam(ptrNet.parameters())

program_starts = time.time()
for epoch in range(EPOCHS):
  train(ptrNet, optimizer, epoch + 1)
  evaluateWordSort(ptrNet, epoch + 1)

now = time.time()
print("It has been {0} seconds since the loop started".format(now - program_starts))
