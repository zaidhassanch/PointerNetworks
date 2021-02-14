import os
from cleanFileCode import cleanFile

filePath = "/home/zaid/DrPascal/data/NOVELS/"
files = os.listdir(filePath)

print(files)
outFile = open("outCombined.txt", mode='w')

for fileName in files:
    cleanFile(filePath, fileName, outFile)

outFile.close()

#=========================================================================
# def combineFiles(filePath):
#
#     filenames = os.listdir(filePath)
#     with open('combined.txt', 'w') as outfile:
#         for fname in filenames:
#             with open(filePath+fname) as infile:
#                 for line in infile:
#                     outfile.write(line)

