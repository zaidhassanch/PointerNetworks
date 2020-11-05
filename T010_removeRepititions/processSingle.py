import random
import spacy
from dataGen.generateData import randomizeSentence, prepareInputForPtrNet, makeSentenceDict
import config
from pointerNetwork import PointerNetwork
import torch
import torch.nn as nn

def modelEvaluateSingle(x_val, y_val, text_val, ptrNet):
  
  """Evaluate after a train epoch"""
  #print('Epoch [{}] -- Evaluate'.format(epoch))

  #x_val, y_val, text_val = getBatch(sentenceData, 8)
  out, _ = ptrNet(x_val, y_val, teacher_force_ratio=0., train=False)
  out = out.permute(1, 0)

  outSent = ""
  inputSent = ""

  for i in range(out.size(0)):
    # print("=============================================")
    #print("yref", y_val[i], out[i], y_val[i] - out[i])
    print("yref", y_val[i], out[i], y_val[i] - out[i])

    v = torch.Tensor.cpu(out[i]).numpy()
    # print("Outpt   Sentence:  ", end="")
    for index in v:
      outSent += text_val[i][index]+" "
      

    for index in range(len(v)):
      inputSent += text_val[i][index]+" "
      

  return inputSent, outSent  

def processSentence(nlp, model, jumbled_sentence):
  success, sentDict = makeSentenceDict(nlp, jumbled_sentence)
  augmentedSentence = randomizeSentence(sentDict["wordArray"])

  x, y, text = prepareInputForPtrNet(augmentedSentence)
  
  if(config.GPU == True):
    xx = torch.cuda.FloatTensor([x])
    yy = torch.cuda.LongTensor([y])
  else:
    xx = torch.FloatTensor([x])
    yy = torch.LongTensor([y])

  # print(text, y)
  sent = modelEvaluateSingle(xx,yy, [text], model)
  return sent



def processSingle(path):
  if config.GPU == True:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE).cuda()
  else:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE)

  ptrNet.load_state_dict(torch.load(path))


  nlp = spacy.load("en_core_web_sm") 
  sentences  =[
    "this is definitely a difficult sentence",
    "your plan seems excellent",
    "it is really working",
    "let me start with a poem",
    "he is sure to pass the exam",
    "we meet every wednesday",
    "do plan to we it tomorrow",
    "i like working on brain",
    "development good is a this",
    "i am going to canada",
    "The script that I am describing here mainly for training and validation "]

  if True:
    for origLine in sentences:
      print("")
      inputLine, outputLine = processSentence(nlp, ptrNet, origLine)
      print("Input Sentence: ", inputLine)
      print("Outpt Sentence: ", outputLine)
      print("Orig  Sentence: ", origLine)

  else:
    lines = open("../data/englishSentences_test.dat").read().splitlines()
    for i in range(7):
      print("\n")
      origLine = random.choice(lines)
      inputLine, outputLine = processSentence(nlp, ptrNet, origLine)
      print("Input Sentence: ", inputLine)
      print("Outpt Sentence: ", outputLine)
      print("Orig  Sentence: ", origLine)

    
processSingle("state_dict_model.pt")
#https://code.visualstudio.com/docs/remote/remote-overview
#https://www.youtube.com/watch?v=QW70p8lLE4A


