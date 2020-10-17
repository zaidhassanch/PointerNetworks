import torch
import spacy
import random
import numpy as np
import random
from main3 import getGoodSentences

nlp = spacy.load("en_core_web_sm")  # make sure to use larger model!
sents = getGoodSentences()
print(sents[4])

def generateSentence1(sentenceLength = 8):
    length = len(sents[sentenceLength])
    index = random.randint(0,length-1)
    # sentenceLength = 8;
    # index = 2;
    sentence = sents[sentenceLength][index];
    #sentence = 'The quick brown dog'#%['The', 'quick', 'brown', 'fox']

    #sentence = "It's my fault that the cake was burned"
    s = sentence.split(" ");
    #print(len(s))
    tokens = nlp(sentence)
    return tokens


def generateSentence(sentenceLength = 8):

    tokens = []
    while(len(tokens) != sentenceLength):
        tokens = generateSentence1(sentenceLength=8)
    if(len(tokens) != sentenceLength):
        x  = 3
    # print(len(tokens))

    return tokens

# x = generateSentence()
# print(x)
# exit()

def randomizeSentence():
    sentence = generateSentence(8)
    augmentedSentence = []
    count = 0
    for word in sentence:
        augmentedSentence.append([word, count])
        count += 1

    random.shuffle(augmentedSentence)
    count = 0
    for word in augmentedSentence:
        word.append(count)
        count += 1
    return augmentedSentence

def prepareInputForPtrNet(list):
    origList = list.copy()

    list.sort(key=lambda e: e[1])
    input = [x[0].tensor for x in origList]
    target = [x[2] for x in list]
    text = [x[0].text for x in origList]
    return input, target, text



def batch(batchSize):
    xx = [];
    yy = [];
    tt = [];
    count = 0
    for i in range(batchSize):
        count += 1
        sentence = randomizeSentence()
        # print(sentence)
        x, y, text = prepareInputForPtrNet(sentence)

        xx.append(x)
        yy.append(y)
        tt.append(text)
    # xx = torch.cuda.FloatTensor(xx)
    # yy = torch.cuda.LongTensor(yy)

    xx = torch.FloatTensor(xx)
    yy = torch.LongTensor(yy)

    return xx, yy, tt


sentence = randomizeSentence()
print(sentence)
x, y, text = prepareInputForPtrNet(sentence)
print(x)
print(y)
print(text)

# nlp = spacy.load("en_core_web_sm")  # make sure to use larger model!
# x, y, t = batch(5)
# print(x.shape)
# print(y)
# print(t)
