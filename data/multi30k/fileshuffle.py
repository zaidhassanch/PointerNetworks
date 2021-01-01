# Include standard modules
import os
import argparse
import random
parser = argparse.ArgumentParser()
parser.add_argument("--infile", "-i", help="the input file")
args = parser.parse_args()
assert(args.infile)


def shuffleLine(line):
    #line = "Four guys three wearing hats one not are jumping at the top of a staircase."
    line = line.replace(".", "")
    line = removeStopWords(line)
    words = line.split()
    random.shuffle(words)
    #print(words)
    line = " ".join(words) + "."
    #print(line)
    #exit()

    return line+"\n"

def removeStopWords(line, stop_words=["a", "an", "the", "at", "are", "were", "is", "on", "in", "with", "of", "and", "for"]):
    #line = "Four guys three wearing hats one not are jumping at the top of a staircase."
    #line = line.replace(".", "")
    line = line.lower()
    if(line[:2] == 'a '):
        line = line[2:]
    if(line[:3] == 'an '):
        line = line[3:]
    if (line[:4] == 'the '):
        line = line[4:]


    for s_word in stop_words:
        line = line.replace(" " + s_word + " ", " ")
    # words = line.split()
    #random.shuffle(words)
    #print(words)
    #line = " ".join(words) + "."
    #print(line)
    #exit()
    return line


line = "Four guys three wearing hats one not are jumping at the top of a staircase."
print(shuffleLine(line))
# exit()
outfile = args.infile  + "nsw"
fw = open(outfile, 'w')

count = 0
with open(args.infile) as fp:
    for line in fp:
        count += 1
        fw.write(shuffleLine(line))
        print(line)

fw.close()

