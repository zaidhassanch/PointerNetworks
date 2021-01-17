# Include standard modules
import os
import pickle
import argparse
import random
parser = argparse.ArgumentParser()
parser.add_argument("--infile", "-i", help="the input file")
args = parser.parse_args()
assert(args.infile)

MAX_LENGTH = 15

pkl_file = open('BPE/data.pkl', 'rb')
data1 = pickle.load(pkl_file)
sp_gec = data1["sp_gec_orig"]

line = "Four guys three wearing hats one not are jumping at the top of a staircase."
print(line)

infilePath = "/data/chaudhryz/ankit/" + args.infile
# exit()
outfile = "shortest" + args.infile
fw = open(outfile, 'w')

count = 0
writtenLines = 0
with open(infilePath) as fp:
    for line in fp:
        count += 1
        splitLine = line.split("\t")

        src = splitLine[0]
        trg = splitLine[1]
        bow = splitLine[2]

        src_encoded = sp_gec.encode(src)
        trg_encoded = sp_gec.encode(trg)
        bow_encoded = sp_gec.encode(bow)

        src_encoded_len = len(src_encoded)
        trg_encoded_len = len(trg_encoded)
        bow_encoded_len = len(bow_encoded)

        if count%4000 == 0:
            print(count, src)
        if src_encoded_len < MAX_LENGTH and trg_encoded_len < MAX_LENGTH and bow_encoded_len < MAX_LENGTH:
            if src_encoded_len > 6 and trg_encoded_len > 6:
                if False == any(char.isdigit() for char in line):
                    fw.write(line)
                    writtenLines += 1
                    if writtenLines == 29000:
                        break
        

fw.close()


