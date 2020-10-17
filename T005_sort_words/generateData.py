import config
from readWords import readInWords
import random

word_list = readInWords();
random.shuffle(word_list)

print(word_list[0:5])



# first stick to five words
for i in range(config.DATASET_SIZE):
	seqLength = 5
	#words = ['ab', 'zac', 'abc', 'f', 'e', 'd'];
	wordsSelected = random.sample(word_list, seqLength)
	print(wordsSelected)
	
	nList =[];

	count = 0
	for word in wordsSelected:
		nList.append([word, count])
		count += 1;

	print(nList)


	nList.sort(key=lambda e: e[0])

	print(nList)

	exit()

	
