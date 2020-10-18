import torch
import random
import numpy as np
import random
import config
from filterSentences import sents, nlp

def generateSentence1(sentenceLength = 8, newSentences = False):
    length = len(sents[sentenceLength])
    index = random.randint(0,length-1)
    sentence = sents[sentenceLength][index];
    tokens = nlp(sentence)
    return tokens

def generateSentence(sentenceLength = 8, newSentences = False):
    tokens = []
    while(len(tokens) != sentenceLength):
        tokens = generateSentence1(sentenceLength, sentenceLength)
    if(len(tokens) != sentenceLength):
        print("Unexpected case found")
        exit()

    return tokens

# x = generateSentence()
# print(x)
# exit()

def randomizeSentence(sentenceLength, newSentences = False):
    sentence = generateSentence(sentenceLength, newSentences)
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



def batch(batchSize, newSentences = False):
    xx = [];
    yy = [];
    tt = [];
    count = 0
    sentenceLength = random.randint(4,10)
    sentenceLength = 6 
    for i in range(batchSize):
        count += 1
        sentence = randomizeSentence(sentenceLength, newSentences)
        # print(sentence)
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


# sentence = randomizeSentence()
# print(sentence)
# x, y, text = prepareInputForPtrNet(sentence)
# print(x)
# print(y)
# print(text)

# nlp = spacy.load("en_core_web_sm")  # make sure to use larger model!
# x, y, t = batch(5)
# print(x.shape)
# print(y)
# print(t)
