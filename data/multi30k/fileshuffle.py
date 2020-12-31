# Include standard modules
import os
import argparse
import random
import spacy

parser = argparse.ArgumentParser()
parser.add_argument("--infile", "-i", help="the input file")
args = parser.parse_args()
assert(args.infile)

spacy_eng = spacy.load("en")

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def sortPos(list):
    newList = list.copy()
    newList.sort(key=lambda e: e[1])
    return newList

def appendPos(list):
    nList = []
    count = 0
    for word in list:
        nList.append([word,count])
        count += 1
    return nList

def appendPosPos(list):
    nList = []
    count = 0
    for word in list:
        word.append(count)
        nList.append(word)
        count += 1
    return nList

def shuffleLine(line):
    #line = "Four guys three wearing hats one not are jumping at the top of a staircase."
    line = line.replace("\n", "")

    tokenized_line = tokenize_eng(line)
    print("tokenized sentence:", tokenized_line)

    pos_tokenized_line = appendPos(tokenized_line)
    print("position appended tokenized sentence:", pos_tokenized_line)

    # words = line.split()
    random.shuffle(pos_tokenized_line)
    pos_pos_tokenized_line = appendPosPos(pos_tokenized_line)
    print("(position appended)^2 tokenized sentence:", pos_pos_tokenized_line)

    sorted_pos_pos_tokenized_line = sortPos(pos_pos_tokenized_line)

    print("(position appended)^2 tokenized sentence:", sorted_pos_pos_tokenized_line)

    shuffleWordsArray = [item[0] for item in pos_tokenized_line]
    shufflePosArray = [str(item[2]) for item in sorted_pos_pos_tokenized_line]
    #print(words)
    reconstructedLine = " ".join(shuffleWordsArray)
    reconstructedPositions = " ".join(shufflePosArray)
    #print(line)
    #exit()
    tokenized_rec_line = tokenize_eng(reconstructedLine)
    if len(tokenized_line) != len(tokenized_rec_line):
        print("Tokenizations do not have same length")
    return (reconstructedLine + "\n", reconstructedPositions + "\n")

print(shuffleLine("Hi"))
# exit()

line = "Four guys three wearing hats one not are jumping at the top of a staircase."
print(shuffleLine(line))

outfileLines = args.infile  + "s"
outfilePos = (args.infile).split(".")[0]  + ".idx"



fwLines = open(outfileLines, 'w')
fwPos = open(outfilePos, 'w')

count = 0
with open(args.infile) as fp:
    for line in fp:
        count += 1
        wordLine, posLine = shuffleLine(line)
        fwLines.write(wordLine)
        fwPos.write(posLine)
        print(line)

fwLines.close()
fwPos.close()

