
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

args = object();
infile = "test10k.tsv"
# lines = 5

count = 0
fp = open(infile)
n_lines = len(open(infile).readlines())
print(n_lines)
# exit()
fw_src = open("in.src", 'w')
fw_src = open("out.src", 'w')
fw_tgt = open("out.tgt", 'w')
fw_con = open("out.con", 'w')

count = 0
for line in fp:
    count += 1
    print("==========", count, end='')

    print(line)
    linearr = line.split("\t")
    if(len(linearr)<3):
        print("issue found", count)
        print(len(linearr))
        exit()

    fw_src.write(linearr[0]+"\n")
    fw_tgt.write(linearr[1]+"\n")
    fw_con.write(linearr[2]+"\n")

fp.close()
fw_src.close()
fw_tgt.close()
fw_con.close()
