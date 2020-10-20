
import torch
from dataGen.generateData import randomizeSentence, prepareInputForPtrNet
import random
import config



def batch(sentenceData, batchSize):
    sentenceLength = random.randint(4,10)
    sentVectN = sentenceData[sentenceLength]
    length = len(sentVectN)
    xx = []
    yy = []
    tt = []
    for i in range(batchSize):
        index = random.randint(0,length-1)

        sentence = sentVectN[index]
        #print(index, sentence["text"])
        sentence = randomizeSentence(sentence["wordArray"])
        x, y, text = prepareInputForPtrNet(sentence)

        xx.append(x)
        yy.append(y)
        tt.append(text)
    if config.GPU == True:
        xx = torch.cuda.FloatTensor(xx)
        yy = torch.cuda.LongTensor(yy)
    else:
        xx = torch.FloatTensor(xx)
        yy = torch.LongTensor(yy)

    return xx, yy, tt

