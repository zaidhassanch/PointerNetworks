import config
import torch

def featurify(inputList):
	N = config.MAX_LENGTH
	f_extracted = []
	for x in inputList:
		asciiX = [ord(char) for char in x]
		#asciiX += [200] * (N - len(asciiX))
		#print(asciiX)
		f_extracted.append(asciiX)
	return f_extracted





