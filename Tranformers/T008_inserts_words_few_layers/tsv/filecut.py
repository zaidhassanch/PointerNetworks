# Include standard modules
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--infile", "-i", help="the input file")
parser.add_argument("--outdir", "-o", help="the output directory")
parser.add_argument("--lines", "-l", help="lines to retain")
args = parser.parse_args()

# print(args.infile)
assert(args.infile)
assert(args.lines)
assert(args.outdir)

if not os.path.exists(args.outdir):
    print(f"Output directory :{args.outdir}: does not exist")
    exit()

if args.infile:
    n_lines = len(open(args.infile).readlines())
    print(f"Input file is:{args.infile} with lines:{n_lines}")
    if (n_lines < int(args.lines)):
        print(f"\n number of lines {n_lines} in file are less than desired: {args.lines}")

inFileName = (args.infile).split("/")[-1]
print(inFileName)
outfile = args.outdir + "/" + inFileName
fw = open(outfile, 'w')
count = 0
with open(args.infile) as fp:
    for line in fp:
        count += 1
        fw.write(line)
        if(count == int(args.lines)):
            print(f">>>>>>> output lines:{args.lines} written successfully to {outfile}")
            break
fw.close()

