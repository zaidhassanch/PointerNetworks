import torch.optim as optim
from dataGen.dataGenerator import batch, convertToWordSingle, convertToWordsBatch
from transfomer import Transformer

#from data import batch
import config
import time
from pointerNetwork import PointerNetwork
import torch
import torch.nn as nn

BATCH_SIZE = 32
# STEPS_PER_EPOCH = 500
EPOCHS = 50
STEPS_PER_EPOCH = 500


def train(pNet, optimizer, epoch, clip=1.):
  """Train single epoch"""
  print('Epoch [{}] -- Train'.format(epoch))
  criterion = nn.CrossEntropyLoss()
  for step in range(STEPS_PER_EPOCH):
    optimizer.zero_grad()

    # Forward
    x, y = batch(1)
    w = convertToWordsBatch(x)

    output = pNet(x, y)
    best_guess = output.argmax(2).transpose(0,1)
    # [-1, :].item()
    output = output.reshape(-1, output.shape[2])
    target = y.reshape(-1)

    print(target, best_guess)


    optimizer.zero_grad()

    loss = criterion(output, target)

    # Backward
    loss.backward()
    nn.utils.clip_grad_norm_(pNet.parameters(), clip)
    optimizer.step()

    if (step + 1) % 100 == 0:
      print('Epoch [{}] loss: {}'.format(epoch, loss.item()))

def evaluateSort(model, epoch):
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


def evaluateWordSort(model, epoch):
  """Evaluate after a train epoch"""
  print('Epoch [{}] -- Evaluate'.format(epoch))

  x_val, y_val = batch(1)

  batch_size, trg_seq_len = y_val.shape

  sentence_tensor = x_val

  #outputs = [30]
  outputs = list(y_val[0][:3].numpy())

  for i in range(trg_seq_len):
    trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
    trg_tensor = trg_tensor.permute(1, 0)

    with torch.no_grad():
      output = model(sentence_tensor, trg_tensor)

    best_guess = output.argmax(2)[-1, :].item()
    outputs.append(best_guess)

    print(outputs, y_val)
  """
  out, _ = model(x_val, y_val)
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
  """

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
embedding_size = 6
src_pad_idx = 2

ptrNet = Transformer(device, embedding_size, src_pad_idx).to(device)


# ptrNet = PointerNetwork(config.HIDDEN_SIZE)
optimizer = optim.Adam(ptrNet.parameters(), lr=0.01)

program_starts = time.time()
for epoch in range(EPOCHS):
  train(ptrNet, optimizer, epoch + 1)
  evaluateWordSort(ptrNet, epoch + 1)

now = time.time()
print("It has been {0} seconds since the loop started".format(now - program_starts))
