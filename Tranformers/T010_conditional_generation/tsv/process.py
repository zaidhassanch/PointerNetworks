
# Include standard modules
# import os
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--infile", "-i", help="the input file")
# parser.add_argument("--outdir", "-o", help="the output directory")
# parser.add_argument("--lines", "-l", help="lines to retain")
# args = parser.parse_args()

# print(args.infile)
# assert(args.infile)
# assert(args.lines)
# assert(args.outdir)
import numpy as np

args = object();
MAX_LENGTH = 20
infile = "test10k.tsv"

src_lengths = np.zeros((20,))
tgt_lengths = np.zeros((20,))

# lines = 5

count = 0
fp = open(infile)
n_lines = len(open(infile).readlines())
print(n_lines)
# exit()
fw_src = open("shortout10k.src", 'w')
fw_tgt = open("shortout10k.tgt", 'w')
fw_con = open("shortout10k.con", 'w')

count = 0
for line in fp:


    #print(line)
    linearr = line.split("\t")
    if(len(linearr)<3):
        print("issue found", count)
        print(len(linearr))
        exit()

    len_src = len(linearr[0].split(" "))
    len_tgt = len(linearr[1].split(" "))
    len_con = len(linearr[2].split(" "))

    bucket_src = np.int32(len_src / 10)
    bucket_tgt = np.int32(len_tgt / 10)

    src_lengths[bucket_src] += 1
    tgt_lengths[bucket_tgt] += 1

    if len_src > MAX_LENGTH or len_tgt > MAX_LENGTH or len_con > MAX_LENGTH:
        print("alert")
    else:
        count += 1
        print("==========", count, end='')
        fw_src.write(linearr[0]+"\n")
        fw_tgt.write(linearr[1]+"\n")
        fw_con.write(linearr[2])

print(count)

print("src_lengths", src_lengths/np.sum(src_lengths))
print("tgt_lengths", tgt_lengths/np.sum(tgt_lengths))

fp.close()
fw_src.close()
fw_tgt.close()
fw_con.close()
