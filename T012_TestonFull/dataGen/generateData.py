import random

#random.seed(50)

#is this sentence difficult definitely a

def filterSentences(fileName):
    sentences = [];
    for i in range(0, 20):   #max 20length sentences
        sentences.append([]);
    f = open(fileName, "r")
    line = ""
    count = 0
    fileLines = f.readlines()
    print(len(fileLines))
    prevLine = ""
    for line in fileLines:

        engLine  = line;
        tempArr = engLine.split(" ")
        sentences[len(tempArr)].append(engLine)
        count += 1
    return sentences

def printSDict(sentenceDict):
    sentenceLength = len(sentenceDict["wordArray"])
    print(str(sentenceLength) + " ==== ", sentenceDict["text"])
    for v in sentenceDict["wordArray"]:
        print(v["word"], end = " => [")
        for i in range(5):
            print(str(v["vector"][i])+", ", end="")
        print("]")
    print()


def makeSentenceDict(nlp, sentence):
    sentDict = dict()
    sentence = sentence.strip();
    sentenceArray = sentence.split(" ");
    
    tokens = [];
    for word in sentenceArray:
        tokens.append(nlp(word))

    #print(sentence)

    if(len(tokens) != len(sentenceArray)):
        print(len(tokens))
        print(len(sentenceArray))
        print("Unexpected case found #", sentence, "#") # e.g. cannot
        print("==")
        for token in tokens:
            print(token)
        print("==")
        for word in sentenceArray:
            print(word)
        print("==")
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

def prepareDataVect(nlp, sents):
    svect = []
    fc = 0
    for s in sents:
        fc += 1
        nSentences = len(s);
        sentVectN = []
        for i in range(nSentences):
            sentence = s[i]
            success, sentenceDict = makeSentenceDict(nlp, sentence)
            if success==False: continue
            sentVectN.append(sentenceDict)
            if i%30==29: 
                print(fc, nSentences, i)
                # break
        svect.append(sentVectN)
    return svect


def randomizeSentence(sentence):
    augmentedSentence = []
    count = 0


    for word in sentence:
        augmentedSentence.append([word, count])
        count += 1
    # print(augmentedSentence)

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

# sent = ["is", "this", "sentence", "difficult", "definitely", "a"]

# s2 = randomizeSentence(sent)
# print(s2)

