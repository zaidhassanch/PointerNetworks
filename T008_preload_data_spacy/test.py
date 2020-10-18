import torch.optim as optim
from batch import batch
from generateData import prepareDataVect
import config
import time
from pointerNetwork import PointerNetwork
import torch
import torch.nn as nn
import time
import pickle

BATCH_SIZE = 2048
EPOCHS = 10
STEPS_PER_EPOCH = 500

def train(pNet, optimizer, epoch, clip=1.):
  """Train single epoch"""
  print('Epoch [{}] -- Train'.format(epoch))
  # x, y, t = batch(BATCH_SIZE)
  start = time.time()
  for step in range(STEPS_PER_EPOCH):
    optimizer.zero_grad()
    x, y, t = batch(sentenceData, BATCH_SIZE)
      
    # Forward
    out, loss = pNet(x, y)
    # Backward
    loss.backward()
    nn.utils.clip_grad_norm_(pNet.parameters(), clip)
    optimizer.step()
    if (step + 1) % 10 == 0:
      duration = time.time() - start
      print('Epoch [{}] loss: {}  time:{:.2f}'.format(epoch, loss.item(), duration))
      start = time.time()

def evaluateWordSort(model, epoch):
  """Evaluate after a train epoch"""
  print('Epoch [{}] -- Evaluate'.format(epoch))

  x_val, y_val, text_val = batch(sentenceData, 8)
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


def modelTrainAndSave(path):
  if config.GPU == True:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE).cuda()
  else:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE)
  optimizer = optim.Adam(ptrNet.parameters())

  program_starts = time.time()
  for epoch in range(EPOCHS):
    train(ptrNet, optimizer, epoch + 1)
    evaluateWordSort(ptrNet, epoch + 1)

  torch.save(ptrNet.state_dict(), path)

  now = time.time()
  print("It has been {0} seconds since the loop started".format(now - program_starts))

def savePickle(fileName):
  sentenceData = prepareDataVect()
  fx = open(fileName, "wb")
  pickle.dump(sentenceData, fx, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(fileName):
  fx = open(fileName, "rb")
  sentenceData = pickle.load(fx)
  return sentenceData



def modelEvaluate(path):
  if config.GPU == True:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE).cuda()
  else:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE)

  ptrNet.load_state_dict(torch.load(path))
  evaluateWordSort(ptrNet, 1)


#savePickle("sentence.pkl")
sentenceData = loadPickle("sentence.pkl")


modelPath = "state_dict_model.pt"

#modelTrainAndSave(modelPath)
modelEvaluate(modelPath)

