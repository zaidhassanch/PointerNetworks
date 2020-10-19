from generateData import prepareDataVect, filterSentences
import pickle
import spacy

def splitFiles(fileName)
	f = open("../data/englishSentences.txt", "r")
	f1 = open("../data/englishSentences_train.dat", "w")
	f2 = open("../data/englishSentences_test.dat", "w")
	f3 = open("../data/englishSentences_validate.dat", "w")

	lines = f.readlines();
	f.close()

	length = len(lines)

	newLines = [];
	for i in range(length):
	  if i%12 < 7:
	    f1.writelines(lines[i])
	  elif i%12 < 10:
	    f2.writelines(lines[i])
	  else:
	    f3.writelines(lines[i])

	f.close()
	f1.close()
	f2.close()
	f3.close()

def savePickle(nlp, sents, fileName):
  sentenceData = prepareDataVect(nlp, sents)
  fx = open(fileName, "wb")
  pickle.dump(sentenceData, fx, protocol=pickle.HIGHEST_PROTOCOL)


def createPickles():
  nlp = spacy.load("en_core_web_sm")  # make sure to use larger model!
  sents = filterSentences("../data/englishSentences_train.dat")
  savePickle(nlp, sents, "../data/englishSentences_train.pkl")

  sents = filterSentences("../data/englishSentences_test.dat")
  savePickle(nlp, sents, "../data/englishSentences_test.pkl")

  sents = filterSentences("../data/englishSentences_validate.dat")
  savePickle(nlp, sents, "../data/englishSentences_validate.pkl")


splitFiles()
createPickles()
