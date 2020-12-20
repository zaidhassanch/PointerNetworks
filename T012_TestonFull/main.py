import torch.optim as optim
from batch import batch, completeBatch
import config
import time
from pointerNetwork import PointerNetwork
import torch
import torch.nn as nn
import time
import pickle

BATCH_SIZE = 32
EPOCHS = 100
STEPS_PER_EPOCH = 300

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


def computeSentenceAccuracy(accStats, text, v_ref, v_out):

    sentenceLength = len(v_ref);
    
    #accStats = accuracyStats[sentenceLength];
    accStats["sentenceCount"] += 1;
    
    # print("DIFF = [", end="")
    sentenceMatch = True
    for index in range(sentenceLength):
      accStats["wordCount"] += 1;
      refTxt = text[v_ref[index]];
      outTxt = text[v_out[index]];
      if( refTxt == outTxt):
        flag = 0;
        accStats["correctWords"] += 1;
      else:
        flag = 1
        sentenceMatch = False

      # print(str(flag)+" ", end="")
      #print(+" ", end="")
    # print("]")

    if(sentenceMatch == True): accStats["correctSentences"] += 1;
    

def printStats(i, stats):
  if (stats["wordCount"]) != 0:
    wordAccuracy = stats["correctWords"]/stats["wordCount"]*100 
    sentenceAccuracy = stats["correctSentences"]/stats["sentenceCount"]*100
    print('[{}] wordAcc [{:.2f}] SentenceAcc: {:.2f}  sentences = {}'\
      .format(i, wordAccuracy, sentenceAccuracy,stats["sentenceCount"]))

def printAccStats(accuracyStats):
  
  print("====================================================")
  for i in range(20):
    stats = accuracyStats[i]
    wordAccuracy = 0.0
    sentenceAccuracy = 0.0
    
    printStats(i, stats)
      
  print("====================================================")

def compareBatchAccuracy(accuracyStats, text_in, y_ref, y_out):
  stat = {
      "sentenceCount" : 0,
      "wordCount"     : 0,
      "correctSentences": 0,
      "correctWords" : 0
  }
  for i in range(y_out.size(0)):
    # print("=============================================")
    # print("yref", y_ref[i], y_out[i], y_ref[i] - y_out[i])

    # print("input", text_in[i])

    # print("orig", text_in[y_ref[i]])
    v_out = torch.Tensor.cpu(y_out[i]).numpy()
    v_ref = torch.Tensor.cpu(y_ref[i]).numpy()

    # print("ORIG = [", end="")
    # for index in v_ref:
    #   print(text_in[i][index]+" ", end="")
    # print("]")


    # print("OUR = [", end="")
    # for index in v_out:
    #   print(text_in[i][index]+" ", end="")
    # print("]")

    text = text_in[i];
    

    computeSentenceAccuracy(stat, text, v_ref, v_out)
  length = len(text)
  statX = accuracyStats[length]
  statX["sentenceCount"] += stat["sentenceCount"]
  statX["wordCount"] += stat["wordCount"]
  statX["correctSentences"] += stat["correctSentences"]
  statX["correctWords"] += stat["correctWords"]

  printStats(length, stat)
  

  #
def evaluateWordSort(accuracyStats, model, sentenceLength):
  """Evaluate after a train epoch"""
  print('Epoch [{}] -- Evaluate'.format(sentenceLength))

  x_val, y_ref, text_in = completeBatch(sentenceData, sentenceLength)
  y_out, _ = model(x_val, y_ref, teacher_force_ratio=0., train=False)
  y_out = y_out.permute(1, 0)
  compareBatchAccuracy(accuracyStats, text_in, y_ref, y_out)

def modelTrain(accuracyStats, PATH):
  if config.GPU == True:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE).cuda()
  else:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE)
  optimizer = optim.Adam(ptrNet.parameters())

  program_starts = time.time()
  for epoch in range(EPOCHS):
    train(ptrNet, optimizer, epoch + 1)
    evaluateWordSort(accuracyStats, ptrNet, epoch + 1)
  # Save
  printAccStats(accuracyStats);

  torch.save(ptrNet.state_dict(), PATH)

  now = time.time()
  print("It has been {0} seconds since the loop started".format(now - program_starts))

def modelEvaluate(accuracyStats, path):
  if config.GPU == True:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE).cuda()
  else:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE)

  ptrNet.load_state_dict(torch.load(path))
  for sentenceLength in range(4,11):
    print(sentenceLength)
    evaluateWordSort(accuracyStats, ptrNet, sentenceLength)
  printAccStats(accuracyStats);


def loadPickle(fileName):
  fx = open(fileName, "rb")
  sentenceData = pickle.load(fx)
  return sentenceData



accuracyStats = []
for i in range(20):
  accuracyStats.append(
    {"sentenceCount" : 0,
    "wordCount"     : 0,
    "correctSentences": 0,
    "correctWords" : 0}
    )
#createPickles()
modelPath = "state_dict_model.pt"

# sentenceData = loadPickle("../data/englishSentences_train.pkl")
# modelTrain(accuracyStats, modelPath)

sentenceData = loadPickle("../data/englishSentences_test.pkl")
modelEvaluate(accuracyStats, modelPath)
