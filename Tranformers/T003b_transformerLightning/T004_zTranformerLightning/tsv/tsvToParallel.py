file = "train4m"
inFile = "/home/chaudhryz/scribendigec/Final/Exp14/" +file + ".tsv"
srcFile = "/home/chaudhryz/PointerNetworks/Tranformers/T003b_transformerLightning/T004_zTranformerLightning/dataFiles/" +file + ".src"
tgtFile = "/home/chaudhryz/PointerNetworks/Tranformers/T003b_transformerLightning/T004_zTranformerLightning/dataFiles/" +file + ".tgt"

fwSRC = open(srcFile, 'w')
fwTGT = open(tgtFile, 'w')

count = 0

with open(inFile, 'r') as fp:
    for line in fp:
        line = line.rstrip()


        if line[0] == "\t":
            line = ["", ""]
        else:
            line = line.split('\t')
        count += 1
        if count%2000 == 1999:
            print("===========")
            print(count)
            print("line[0]", line[0])
            print("line[1]", line[1])

        fwSRC.write(line[0] + "\n")
        fwTGT.write(line[1] + "\n")
