import torch.optim as optim
from dataGen.dataGenerator import batch, convertToWordSingle, convertToWordsBatch
from transfomer import Transformer

#from data import batch
import config
import time
from pointerNetwork import PointerNetwork
import torch
import torch.nn as nn
torch.manual_seed(0)


BATCH_SIZE = 32
# STEPS_PER_EPOCH = 500
EPOCHS = 50
STEPS_PER_EPOCH = 500


def train(model, optimizer, epoch, clip=1.):
  """Train single epoch"""
  model.train()
  print('Epoch [{}] -- Train'.format(epoch))
  criterion = nn.CrossEntropyLoss()

  count = 0
  for step in range(STEPS_PER_EPOCH):
    # print(count)

    optimizer.zero_grad()
    # Forward
    inp_data, target = batch(32)
    # w = convertToWordsBatch(x)



    #output = output.reshape(-1, output.shape[2])
    #target = target[1:].reshape(-1)
    # if(count == 3):
    #   target = target[:3]

    trg = target[:-1, :]
    # trg = target
    output = model(inp_data, trg)
    best_guess = output.argmax(2).transpose(0,1)
    # [-1, :].item()
    output = output.reshape(-1, output.shape[2])
    target = target[1:].reshape(-1)
    # target = target.reshape(-1)

    # if count == 3:
    #   print(count, "...............test")
    #   print(target, best_guess, best_guess-target)
    #   count += 1

    # output = output.reshape(-1, output.shape[2])
    # target = target[1:].reshape(-1)
    # optimizer.zero_grad()
    loss = criterion(output, target)

    # Backward
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    if (step + 1) % 100 == 0:
      count += 1
      print('Epoch [{}] loss: {}'.format(epoch, loss.item()))
#
# def evaluateSort(model, epoch):
#   """Evaluate after a train epoch"""
#   print('Epoch [{}] -- Evaluate'.format(epoch))
#   x_val, y_val = batch(4)
#
#   out, _ = model(x_val, y_val, teacher_force_ratio=0.)
#   out = out.permute(1, 0)
#   sumVal = x_val.sum(dim=2)
#   for i in range(out.size(0)):
#     print('{} --> {} --> {} --> {}'.format(
#       sumVal[i],
#       sumVal[i].gather(0, out[i]),
#       sumVal[i].gather(0, y_val[i]),
#       sumVal[i].gather(0, y_val[i]) - sumVal[i].gather(0, out[i])
#     ))


def evaluateWordSort(model, epoch):
  """Evaluate after a train epoch"""
  print('Epoch [{}] -- Evaluate'.format(epoch))

  model.eval()
  x_val, y_val = batch(1)

  trg_seq_len, batch_size = y_val.shape

  sentence_tensor = x_val

  outputs = [[30]]
  # outputs = list(y_val[0][0].numpy())

  for i in range(trg_seq_len-1):
    trg_tensor = torch.LongTensor(outputs).to(device)
    # trg_tensor = trg_tensor.permute(1, 0)

    with torch.no_grad():
      output = model(sentence_tensor, trg_tensor)

    best_guess = output.argmax(2)[-1, :].item()
    outputs.append([best_guess])

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

ptrNet = Transformer(device, embedding_size, src_pad_idx = src_pad_idx).to(device)


# ptrNet = PointerNetwork(config.HIDDEN_SIZE)
optimizer = optim.Adam(ptrNet.parameters(), lr=0.01)

program_starts = time.time()
for epoch in range(EPOCHS):

  evaluateWordSort(ptrNet, epoch + 1)

  train(ptrNet, optimizer, epoch + 1)
  evaluateWordSort(ptrNet, epoch + 1)

now = time.time()
print("It has been {0} seconds since the loop started".format(now - program_starts))
