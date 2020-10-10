import string
import random
import config

def generateWords():
	alphabets = list(string.ascii_lowercase)

	print(alphabets)

	def genWord():
		strWord = ''
		for i in range(config.WORD_LENGTH):
			a = random.choice(alphabets)
			#print(a, strWord + a)
			strWord = strWord + a
		return strWord	
		


	wordList = []
	for i in range(config.DICT_SIZE):
		word = genWord()

		wordList.append(word)
	#print(wordList)
	#print(len(wordList))

	return wordList

