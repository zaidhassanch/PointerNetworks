from generateData import prepareDataVect, filterSentences
import pickle
import spacy

def splitFiles(fileName):
	f  = open(fileName +  ".txt", "r")
	f1 = open(fileName + "_train.dat", "w")
	f2 = open(fileName + "_test.dat", "w")
	f3 = open(fileName + "_validate.dat", "w")

	lines = f.readlines();
	f.close()

	length = len(lines)

	newLines = [];
	for i in range(length):
	  if i%12 < 7:
	    f1.writelines(lines[i].lower())
	  elif i%12 < 10:
	    f2.writelines(lines[i].lower())
	  else:
	    f3.writelines(lines[i].lower())

	f.close()
	f1.close()
	f2.close()
	f3.close()

def savePickle(nlp, sents, fileName):
  sentenceData = prepareDataVect(nlp, sents)
  fx = open(fileName, "wb")
  pickle.dump(sentenceData, fx, protocol=pickle.HIGHEST_PROTOCOL)


def createPickles(fileName):
  nlp = spacy.load("en_core_web_sm")  # make sure to use larger model!
  sents = filterSentences(fileName + "_train.dat")
  savePickle(nlp, sents, fileName + "_train.pkl")

  sents = filterSentences(fileName + "_test.dat")
  savePickle(nlp, sents, fileName + "_test.pkl")

  sents = filterSentences(fileName + "_validate.dat")
  savePickle(nlp, sents, fileName + "_validate.pkl")

fName = "../../data/englishSentences"
splitFiles(fName)
createPickles(fName)
