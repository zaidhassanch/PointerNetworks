import torch.optim as optim
from batch import batch
import config
import time
from pointerNetwork import PointerNetwork
import torch
import torch.nn as nn
import time
import pickle

BATCH_SIZE = 64
EPOCHS = 30
STEPS_PER_EPOCH = 100

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

def compareBatchAccuracy(text_in, y_ref, y_out):
  for i in range(y_out.size(0)):
    print("=============================================")
    print("yref", y_ref[i], y_out[i], y_ref[i] - y_out[i])

    print("input", text_in[i])

    # print("orig", text_in[y_ref[i]])
    v_out = torch.Tensor.cpu(y_out[i]).numpy()
    v_ref = torch.Tensor.cpu(y_ref[i]).numpy()

    print("ORIG = [", end="")
    for index in v_ref:
      print(text_in[i][index]+" ", end="")
    print("]")


    print("OUR = [", end="")
    for index in v_out:
      print(text_in[i][index]+" ", end="")
    print("]")

    print("DIFF = [", end="")
    for index in range(len(v_ref)):
      refTxt = text_in[i][v_ref[index]];
      outTxt = text_in[i][v_out[index]];
      if( refTxt == outTxt):
        flag = 0;
      else:
        flag = 1
      print(str(flag)+" ", end="")
      #print(+" ", end="")
    print("]")


def evaluateWordSort(model, epoch):
  """Evaluate after a train epoch"""
  print('Epoch [{}] -- Evaluate'.format(epoch))

  x_val, y_ref, text_in = batch(sentenceData, 8)
  y_out, _ = model(x_val, y_ref, teacher_force_ratio=0.)
  y_out = y_out.permute(1, 0)
  compareBatchAccuracy(text_in, y_ref, y_out)



def modelTrain(PATH):
  if config.GPU == True:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE).cuda()
  else:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE)
  optimizer = optim.Adam(ptrNet.parameters())

  program_starts = time.time()
  for epoch in range(EPOCHS):
    train(ptrNet, optimizer, epoch + 1)
    evaluateWordSort(ptrNet, epoch + 1)
  # Save
  torch.save(ptrNet.state_dict(), PATH)

  now = time.time()
  print("It has been {0} seconds since the loop started".format(now - program_starts))

def modelEvaluate(path):
  if config.GPU == True:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE).cuda()
  else:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE)

  ptrNet.load_state_dict(torch.load(path))
  evaluateWordSort(ptrNet, 1)


def loadPickle(fileName):
  fx = open(fileName, "rb")
  sentenceData = pickle.load(fx)
  return sentenceData


#createPickles()
modelPath = "state_dict_model_10.pt"

sentenceData = loadPickle("../data/englishSentences_train.pkl")
modelTrain(modelPath)

# sentenceData = loadPickle("../data/englishSentences_test.pkl")
# modelEvaluate(modelPath)
