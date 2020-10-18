import spacy
import random
import config
import torch

def filterSentences():
    sentences = [];
    for i in range(0, 20):   #max 20length sentences
        sentences.append([]);
    f = open("../data/eng-fra.txt", "r")
    line = ""
    count = 0
    fileLines = f.readlines()
    print(len(fileLines))
    prevLine = ""
    for line in fileLines:

        lines = line.split("\t");
        englishLine  = lines[0];
        englishLines = englishLine.split(".")
        if(len(englishLines)>2): continue

        for engLine in englishLines:
            engLine  = engLine.strip()
            if '&' in engLine: continue
            if '"' in engLine: continue
            if ',' in engLine: continue
            if "'" in engLine: continue
            if ":" in engLine: continue
            if "-" in engLine: continue
            if "%" in engLine: continue
            if "?" in engLine: continue
            if "!" in engLine: continue

            if "0" in engLine: continue
            if "1" in engLine: continue
            if "2" in engLine: continue
            if "3" in engLine: continue
            if "4" in engLine: continue
            if "5" in engLine: continue
            if "6" in engLine: continue
            if "7" in engLine: continue
            if "8" in engLine: continue
            if "9" in engLine: continue
            if "cannot" in engLine: continue

            if(engLine == ""): continue
            tempArr = engLine.split(" ")

            if(len(tempArr)>2 and len(tempArr)<13):
                if (engLine == prevLine):
                    continue
                else:
                    prevLine = engLine
                # print(count, engLine)
                sentences[len(tempArr)].append(engLine)
                count += 1
                # if(count == 100): exit()
    return sentences

nlp = spacy.load("en_core_web_sm")  # make sure to use larger model!
sents = filterSentences()


def printSDict(sentenceDict):
    sentenceLength = len(sentenceDict["wordArray"])
    print(str(sentenceLength) + " ==== ", sentenceDict["text"])
    for v in sentenceDict["wordArray"]:
        print(v["word"], end = " => [")
        for i in range(5):
            print(str(v["vector"][i])+", ", end="")
        print("]")
    print()


def makeSentenceDict(sentence):
    sentDict = dict()
    sentenceArray = sentence.split(" ");

    tokens = nlp(sentence)
    if(len(tokens) != len(sentenceArray)):
        print(len(tokens))
        print(len(sentenceArray))
        print("Unexpected case found #", sentence) # e.g. cannot
        return False, None

    sentDict["wordArray"] = []

    for i in range(len(tokens)):
        #x = [1,2,3,4]#tokens[i].vector
        x = tokens[i].vector
        t = tokens[i].text
        sentDict["wordArray"].append({"word":t, "vector":x})

    sentDict["text"] = sentence
    # sentDict["textArray"] = {sentenceArray
    return True, sentDict

def prepareDataVect():
    svect = []
    fc = 0
    for s in sents:
        fc += 1
        nSentences = len(s);
        sentVectN = []
        for i in range(nSentences):
            sentence = s[i]
            success, sentenceDict = makeSentenceDict(sentence)
            if success==False: continue
            sentVectN.append(sentenceDict)
            if i%30==29: 
                print(fc, nSentences, i)
                break
        svect.append(sentVectN)
    return svect


def randomizeSentence(sentence):
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
    input = [x[0]["vector"] for x in origList]
    target = [x[2] for x in list]
    text = [x[0]["word"] for x in origList]
    return input, target, text


# for sentVectN in sentVect:
#     for sentence in sentVectN:
#         pass
# printSDict(sentence)



