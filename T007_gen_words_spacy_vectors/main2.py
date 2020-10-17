
import spacy
import random
# nlp = spacy.load("en_core_web_md")  # make sure to use larger model!

def generateSentence():
    sentence = ['The', 'quick', 'brown', 'fox']
    return sentence

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
    input = [x[0] for x in origList]
    target = [x[2] for x in list]
    return input, target

sentence = randomizeSentence()
print(sentence)
x, y = prepareInputForPtrNet(sentence)
print(x)
print(y)

