import string
import random

def generateWord(minWordLength = 2, maxWordLength = 4):
    word = ''
    alphabets = string.ascii_lowercase
    wordLength = random.randint(minWordLength, maxWordLength)
    for k in range(wordLength):
        word += random.sample(alphabets, 1)[0]
    return word

# print(generateWord())

def generateWords(n, minLength, maxLength):
    x1 = [];
    for i in range(n):
        w = generateWord(minLength, maxLength) #generate two lettered words
        x1.append(w);
    nList = []
    count = 0
    for word in x1:
        nList.append([word, count])
        count += 1
    return nList

def sortWords(list):
    newList = list.copy()
    newList.sort(key=lambda e: e[0])
    return newList

def prepareInputForPtrNet(origList, sortedList):
    input = [x[0] for x in origList]
    target = [x[1] for x in sortedList]
    return input, target

def convertAlphabetsToInts(wordArray):
    outArray = [];
    for word in wordArray:
        x = [];
        for alphabet in word:
            x.append(ord(alphabet))
        outArray.append(x);
    return outArray    

def convertIntsToAlphabets(intArray):
    outArray = []
    for word in intArray:
        x = ''
        for alphabet in word:
            x = x + chr(alphabet)
        outArray.append(x)
    return outArray

