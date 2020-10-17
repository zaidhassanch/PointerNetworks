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
    return torch.FloatTensor(input), target, text

sentence = randomizeSentence()
print(sentence)
x, y, text = prepareInputForPtrNet(sentence)
print(x)
print(y)
print(text)

