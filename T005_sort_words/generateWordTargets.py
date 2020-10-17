import string
import random
import config
from generateWords import generateWords

wordList = generateWords()

print("Hello world")


def genInTargetBatch(batch_size):
	inputs = []
	targets = []
	for i in range(batch_size):

		#words = ['ab', 'zac', 'abc', 'f', 'e', 'd'];
		wordsSelected = random.sample(wordList, config.SEQ_LENGTH)
		#print(wordsSelected)

		nList =[];

		count = 0
		for word in wordsSelected:
			nList.append([word, count])
			count += 1;

		#print(nList)

		originalList = nList.copy()
		nList.sort(key=lambda e: e[0])

		input = [x[0] for x in originalList]
		target = [x[1] for x in nList]

		#print(input)
		#print(target)

		inputs.append(input)
		targets.append(target)

	return inputs, targets
