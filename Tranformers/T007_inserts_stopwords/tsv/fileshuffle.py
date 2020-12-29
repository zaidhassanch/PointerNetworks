# Include standard modules
import os
import argparse
import random
parser = argparse.ArgumentParser()
parser.add_argument("--infile", "-i", help="the input file")
args = parser.parse_args()
assert(args.infile)

def shuffleLine(line):
    line = "Four guys three wearing hats one not are jumping at the top of a staircase."
    line = line.replace(".", "")
    words = line.split()
    random.shuffle(words)
    #print(words)
    line = " ".join(words) + "."
    #print(line)
    #exit()
    return line+"\n"


line = "Four guys three wearing hats one not are jumping at the top of a staircase."
print(shuffleLine(line))
# exit()
outfile = args.infile  + "s"
fw = open(outfile, 'w')

count = 0
with open(args.infile) as fp:
    for line in fp:
        count += 1
        fw.write(shuffleLine(line))
        print(line)

fw.close()

