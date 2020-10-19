
import spacy
from dataGen.generateData import randomizeSentence, prepareInputForPtrNet, makeSentenceDict
import config
from pointerNetwork import PointerNetwork
import torch
import torch.nn as nn

def modelEvaluateSingle(x_val, y_val, text_val, path):
  if config.GPU == True:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE).cuda()
  else:
    ptrNet = PointerNetwork(config.HIDDEN_SIZE)

  ptrNet.load_state_dict(torch.load(path))
  """Evaluate after a train epoch"""
  #print('Epoch [{}] -- Evaluate'.format(epoch))

  #x_val, y_val, text_val = getBatch(sentenceData, 8)
  out, _ = ptrNet(x_val, y_val, teacher_force_ratio=0.)
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

def processSingle():
	modelPath = "state_dict_model.pt"
	nlp = spacy.load("en_core_web_sm") 
	jumbled_sentence = "We so close were"
	success, sentDict = makeSentenceDict(nlp, jumbled_sentence)
	augmentedSentence = randomizeSentence(sentDict["wordArray"])

	x, y, text = prepareInputForPtrNet(augmentedSentence)
	xx = torch.FloatTensor([x])
	yy = torch.LongTensor([y])
	# print(text, y)
	modelEvaluateSingle(xx,yy, [text], modelPath)


processSingle()