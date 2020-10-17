import torch
import spacy
import random
import numpy as np
nlp = spacy.load("en_core_web_sm")  # make sure to use larger model!

def generateSentence():
    sentence = 'The quick brown dog'#%['The', 'quick', 'brown', 'fox']
    tokens = nlp(sentence)

    return tokens

def randomizeSentence():
    sentence = generateSentence()
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
    for i in range(batchSize):
        sentence = randomizeSentence()
        # print(sentence)
        x, y, text = prepareInputForPtrNet(sentence)
        xx.append(x)
        yy.append(y)
        tt.append(text)
    xx = torch.cuda.FloatTensor(xx)
    yy = torch.cuda.LongTensor(yy)
    return xx, yy, tt


# sentence = randomizeSentence()
# print(sentence)
# x, y, text = prepareInputForPtrNet(sentence)
# print(x.shape)
# print(y)
# print(text)

# nlp = spacy.load("en_core_web_sm")  # make sure to use larger model!
# x, y, t = batch(5)
# print(x.shape)
# print(y)
# print(t)
