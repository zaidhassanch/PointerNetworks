import torchtext
from torchtext.data import Field, BucketIterator,TabularDataset
from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_numericalizer, sentencepiece_tokenizer
import nltk
from nltk.translate import bleu_score
import sentencepiece as spm
import dill
import torch
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import sys
import re

lemmatizer = WordNetLemmatizer()

def spm_args(data_path, model_prefix, vocab_size ):
	return ''.join(["--pad_id=3 --bos_id=0 --eos_id=1 --unk_id=2",' --input=', str(data_path)," --model_prefix=", model_prefix, " --vocab_size=",str(vocab_size)," --hard_vocab_limit=false"])

def cleanO(s):
	s = str(s)
	s = re.sub(r'[^A-Za-z0-9,\.!?\';()]', " ", s)
	s = re.sub('[\s]+',' ', s)
	s = s.lstrip().rstrip()
	return s

def tokenizerSW(s): # create a tokenizer function
	s = cleanO(s)
	s = nltk.word_tokenize(s.lower())
	s = [i for i in s if i.isalpha()]
	s = [i for i in s if wordnet.synsets(i)]
	s = [lemmatizer.lemmatize(i) for i in s]
	s = [word for word in s if word not in stopwords.words('english')]
	if not s:
		s = ["na"]
	s = " ".join(s)
	return s

if __name__ == '__main__':
	#2015: 3 columns. orig, edited, editFlag
	#2017: 4 columns. orig, edited, editFlag, numChanges.
	#python BPEModel.py train 2017
	tokenize = lambda x: x.split()
	if sys.argv[1] == "train":
	#Remodel everything so it is part of the main dataset and not the tokenizer.. 'cause nltk is sloooooowwwwwwwwwwwwwwwwwww!! and realtime tokenizing will be a massive bummer.
	#Also add information flags to kinda create the dataset size on the go from the scribendi files..
		dataTyp = sys.argv[2]
		dataTyp = str(dataTyp)
		trainingSz = 300000
		testSz = 10000
		vocab_size = 30000
		if dataTyp == "2017":
			data_path = "../2017-data/"
			fN = "2017.tsv"
			cZ = 4
		else:
			data_path = "../data/"
			fN = "2015.tsv"
			cZ = 3
		S = []
		f = data_path+fN
		sent = pd.read_csv(f, sep="\t", usecols=[0,1,2])
		edited = sent[sent["edited"] == True]
		edited = edited.sample(n = trainingSz+testSz)
		s1 = [cleanO(i) for i in edited["original_sentence"]]
		s2 = [cleanO(i) for i in edited["edited_sentence"]]
		S += s1
		S += s2
		x = open("all_Sent.csv", "w")
		for i in S:
			x.write(i+"\n")
		x.close()
		#create vocab and BPE model.
		GEC_IP = "all_Sent.csv"
		model_ip = spm_args(GEC_IP, "GCEBPE30k", vocab_size)
		spm.SentencePieceTrainer.train(model_ip)
		#Creating Dataset:
		edited = edited.reset_index(drop=True)
		l = len(edited)
		DS = []
		for i in range(l):
			o = cleanO(edited["original_sentence"][i])
			e = cleanO(edited["edited_sentence"][i])
			uE = tokenizerSW(edited["edited_sentence"][i])
			#uE = " ".join(uE)
			DS.append([o, e, uE])
		train = DS[:trainingSz]
		test = DS[trainingSz:]
		#Train
		x = open("train300k.tsv", "w")
		for i in train:
			x.write(i[0]+"\t"+i[1]+"\t"+i[2]+"\n")
		x.close()
		#Test
		x = open("test10k.tsv", "w")
		for i in test:
			x.write(i[0]+"\t"+i[1]+"\t"+i[2]+"\n")
		x.close()
	#Other general stuff:
	sp_gec = load_sp_model("GCEBPE30k.model")
	SRC = Field(use_vocab = False, tokenize = sp_gec.encode, init_token = sp_gec.bos_id(), eos_token = sp_gec.eos_id(), pad_token = sp_gec.pad_id(), batch_first = True)
	noSW = Field(use_vocab = True, tokenize = tokenize, init_token='<sos>', eos_token='<eos>', lower = True)
	tv_datafields = [("orig", SRC), ("correction1", SRC), ("correction2", noSW)]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	trn, tst = TabularDataset.splits(path = ".", train = "train300k.tsv", test = "test10k.tsv", format='tsv', skip_header=False, fields = tv_datafields)
	#trn, tst = TabularDataset.splits(path = "../2017-data/", train = 'trainBPE.tsv', test = "testBPE.tsv", format='tsv', skip_header=False, fields = tv_datafields)
	noSW.build_vocab(trn, tst, min_freq = 1)
	#len(SRC.vocab)
	BATCH_SIZE = 32
	train_iterator, test_iterator = BucketIterator.splits((trn, tst), batch_size = BATCH_SIZE, sort_within_batch = True, sort_key = lambda x : len(x.orig), device = device)
	torch.save(SRC, "SRC.Field", pickle_module = dill)
	torch.save(noSW, "noSW.Field", pickle_module = dill)
	example_id = 584
	print("Original:")
	example_orig = vars(trn.examples[example_id])['orig']
	print(example_orig)
	print(sp_gec.decode(example_orig))
	print("Correction:")
	example_corr = vars(trn.examples[example_id])['correction1']
	print(example_corr)
	print(sp_gec.decode(example_corr))
	print("Correction2:")
	example_orig = vars(trn.examples[example_id])['correction2']
	print(example_orig)
	#print(sp_gec.decode(example_orig))
	print("GEC-BPE Len:")
	print(len(sp_gec))
	print("noSW Vocab Len:")
	print(len(noSW.vocab))