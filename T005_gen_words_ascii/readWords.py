

def readInWords():
	f = open("/usr/share/dict/words", "r")
	char_list = ['\'', 'é', 'ö']

	f = [ele for ele in f if all(ch not in ele for ch in char_list)] 
	f = [ele for ele in f if len(ele) < 14]

	f = [ele.rstrip() for ele in f]
	f = [ele.lower() for ele in f]

	maxlength = 0
	for i, x in enumerate(f):
	  print(x)
	  if maxlength < len(x):
	  	maxlength = len(x)
	  	print(f[i])

	print(maxlength)

	print(type(f))
	return f

def writeWords(words):
	f = open("words.txt", "w")
	for word in words:
		if(len(word) == 6):
			f.write(word + "\n");
	f.close()

f = readInWords()
writeWords(f)
print(f[0:5])
