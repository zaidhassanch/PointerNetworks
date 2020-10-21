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
  out, _ = ptrNet(x_val, y_val, teacher_force_ratio=0.)
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
    "Try to control yourselves",
    "Twenty families live here",
    "Two seats remained vacant",
    "Wait until further notice",
    "Watch out for pickpockets",
    "We agreed among ourselves",
    "We all agreed unanimously",
    "We have people everywhere",
    "We helped him financially",
    "We must leave immediately",
    "We started before sunrise",
    "Wine helps with digestion",
    "Write the address clearly",
    "You are easily distracted",
    "You must control yourself",
    "You seem distracted today",
    "You should eat vegetables",
    "Your English is improving",
    "Your boyfriend looks cute",
    "Your house needs painting",
    "Your plan seems excellent",
    "It is really working",
    "Let me start with a poem",
    "He is sure to pass the exam",
    "We are having a hard time",
  ]

  if True:
    for origLine in sentences:
      print("")
      inputLine, outputLine = processSentence(nlp, ptrNet, origLine)
      print("Input Sentence: ", inputLine)
      print("Outpt Sentence: ", outputLine)
      print("Orig  Sentence: ", origLine)

  else:
    lines = open("../data/englishSentences_test.dat").read().splitlines()
    for i in range(5):
      print("\n")
      origLine = random.choice(lines)
      inputLine, outputLine = processSentence(nlp, ptrNet, origLine)
      print("Input Sentence: ", inputLine)
      print("Outpt Sentence: ", outputLine)
      print("Orig  Sentence: ", origLine)

    
processSingle("state_dict_model.pt")



