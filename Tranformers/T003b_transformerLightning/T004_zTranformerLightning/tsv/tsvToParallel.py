file = "test10k"
inFile = "/home/chaudhryz/scribendigec/Final/Exp14/" +file + ".tsv"
srcFile = "/home/chaudhryz/PointerNetworks/Tranformers/T003b_transformerLightning/T004_zTranformerLightning/dataFiles/" +file + ".src"
tgtFile = "/home/chaudhryz/PointerNetworks/Tranformers/T003b_transformerLightning/T004_zTranformerLightning/dataFiles/" +file + ".tgt"

fwSRC = open(srcFile, 'w')
fwTGT = open(tgtFile, 'w')

count = 0

with open(inFile, 'r') as fp:
    for line in fp:
        line = line.strip()
        if count == 332:
            print("===========")
            print("line[0]:", line[0], ":line[0]")
            print("line[1]:", line[1], ":line[1]")

        if line[0] == "\t":
            line[0] = ""
            line[1] = ""
        else:
            line = line.split('\t')
        count += 1
        if count == 333:
            print("===========")
            print("line[0]", line[0])
            print("line[1]", line[1])
        print(count, line[0])
        fwSRC.write(line[0] + "\n")
        fwTGT.write(line[1] + "\n")
