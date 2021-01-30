import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.data.functional import generate_sp_model, load_sp_model
from queue import Queue
from queue import PriorityQueue
import re
import numpy as np
import random
import math
import time
import pickle
from torchtext.data.metrics import bleu_score
from s2s import Seq2Seq
from torch.autograd import Variable
import random
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.gleu_score import corpus_gleu


def translateData(sentence, src_field):
	tokens = sentence
	#tokens = sp_gec.encode(sentence)
	#tokens = [token.lower() for token in sentence]
	tokens = [src_field.init_token] + tokens + [src_field.eos_token]
	#src_indexes = [src_field.vocab.stoi[token] for token in tokens]
	src_indexes = tokens
	src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
	src_len = torch.LongTensor([len(src_indexes)]).to(device)
	return src_tensor, src_len

#HERE

class Node(object):
	def __init__(self, hidden, previous_node, decoder_input, log_prob, length):
		self.hidden = hidden# 当前单词的hidden_state
		self.previous_node = previous_node
		self.decoder_input = decoder_input
		self.log_prob = log_prob
		self.length = length

class Beam_Search(nn.Module):
	# Sequence to Sequence Learning with Neural Networks
	def __init__(self, model):
		super(Beam_Search, self).__init__()
		#self.sos_idx = sos_idx
		model.eval()
		self.encoder = model.encoder
		self.decoder = model.decoder
	def forward(self, src, src_field, device, max_len = 30):
		sos_idx = src_field.init_token
		eos_idx = src_field.eos_token
		src_tensor, src_len = translateData(src, src_field)
		batch_size = src_tensor.size(1)
		max_len = len(src)+5
		sos_input = [sos_idx for _ in range(batch_size)]
		# decoder_input: [batch_size]
		# all_tokens: [batch_size, 1]
		# all_scores: [batch_size, 1]
		all_tokens = torch.LongTensor(batch_size, max_len).fill_(eos_idx).to(device)
		all_scores = torch.FloatTensor(batch_size, max_len).fill_(0).to(device)
		# encoder_outputs: [seq_len, batch_size, hidden_size]
		# encoder_hidden: [batch_size, hidden_size]
		# decoder_hidden: [batch_size, hidden_size]
		#src_tensor, src_len = translateData(src, src_field)
		with torch.no_grad():
			syntax, content = self.encoder(src_tensor, src_len)
		encoder_hidden = torch.cat((syntax, content), dim = 1)
		#encoder_outputs, encoder_hidden = self.encoder(src)
		decoder_hidden = encoder_hidden
		for batch_idx in range(batch_size):
			# _decoder_input: [1]
			# _encoder_outputs: [seq_len, 1, hidden_size]
			# _decoder_hidden: [1, hidden_size]
			#torch.LongTensor([sos_token]).to(device)
			_decoder_input = torch.LongTensor([sos_idx]).to(device)
			#_encoder_outputs = encoder_outputs[:, batch_idx, :].unsqueeze(1)
			#_decoder_hidden = decoder_hidden[batch_idx, :].unsqueeze(0)
			_decoder_hidden = decoder_hidden[batch_idx, :].unsqueeze(0)
			root = Node(hidden=_decoder_hidden, previous_node=None, decoder_input=_decoder_input, log_prob=torch.FloatTensor([1]), length=1)
			q = Queue()
			q.put(root)
			end_nodes = []
			beam_size = 3
			while not q.empty():
				if len(end_nodes) >= beam_size:
					break
				count = q.qsize()
				candidates = PriorityQueue()
				for idx in range(count):
					node = q.get()
					# _decoder_input: [1]
					# _hidden: [1, hidden_size]
					_decoder_input = node.decoder_input
					_hidden = node.hidden
					_log_prob = node.log_prob
					_length = node.length
					if _decoder_input.item() == eos_idx or _length >= max_len:
						end_nodes.append(node)
						break
					# decoder_output: [1, output_dim]
					# decoder_hidden: [1, hidden_size]
					#decoder_input, decoder_hidden, enc_hidden
					_decoder_output, _decoder_hidden = self.decoder(_decoder_input, _hidden, encoder_hidden)
					softmax = nn.Softmax(dim = 1)
					decoder_score = softmax(_decoder_output)
					values, indices = decoder_score.squeeze(0).topk(beam_size)
					for i in range(beam_size):
						prob = values[i] * _log_prob.item()
						index = indices[i]
						_decoder_input = torch.LongTensor([index]).to(device)
						_len = _length + 1
						new_node = Node(hidden=_decoder_hidden, previous_node=node, decoder_input=_decoder_input, log_prob=prob, length=_len)
						candidates.put((prob, new_node))
				if candidates.qsize() == 0:
					continue
				# 小根堆 转成 大根堆
				# Small root pile turns into big root pile
				_candidates = []
				while not candidates.empty():
					binode = candidates.get()
					_candidates = [binode] + _candidates
				# 取topk = beam_size个元素
				# Take topk = beam_size elements
				for i in range(beam_size):
					binode = _candidates[i]
					q.put(binode[1])
			# 回溯: 找到整个句子的所有节点
			# Backtracking: find all nodes of the entire sentence
			probs, indexs = torch.Tensor([node.log_prob for node in end_nodes]).topk(1)
			max_node = end_nodes[indexs[0]]
			_token_idxs = []
			_token_scores = []
			while max_node is not None:
				_token_idxs = [max_node.decoder_input.item()] + _token_idxs
				_token_scores = [max_node.log_prob.item()] + _token_scores
				max_node = max_node.previous_node
			token_len = len(_token_idxs)
			token_idxs = torch.LongTensor(max_len).fill_(eos_idx)
			token_scores = torch.FloatTensor(max_len).fill_(0)
			token_idxs[:token_len] = torch.LongTensor(_token_idxs).to(device)
			token_scores[:token_len] = torch.FloatTensor(_token_scores).to(device)
			all_tokens[batch_idx] = token_idxs
			all_scores[batch_idx] = token_scores
		#return all_tokens, all_scores
		all_tokens = all_tokens.squeeze().tolist()
		return all_tokens

def beam_search(src, src_field, model, device, max_len = 30):
	max_length = len(src)+5
	model.eval()
	encoder = model.encoder
	decoder = model.decoder
	sos_token = src_field.init_token
	eos_token = src_field.eos_token
	beam_search_k = 3
	#src_tensor: [TxB], src_len: [B]
	src_tensor, src_len = translateData(src, src_field)
	with torch.no_grad():
		syntax, content = encoder(src_tensor, src_len)
	#[BxH]
	enc_hidden = torch.cat((syntax, content), dim = 1)
	decoder_hidden = enc_hidden
	#dec_ip: [B]
	decoder_input = torch.LongTensor([sos_token]).to(device)
	# Hidden layers used to record the k choices with the highest probability, type: [torch.tensor()]
	#[beam x (BxH)]
	hidden_log = [decoder_hidden for _ in range(beam_search_k)]
	# Used to record the largest k probabilities, type: [float]
	prob_log = [0.0 for _ in range(beam_search_k)] # Used to record the decoded output of the k choices with the highest probability, type: [[int]]
	decoder_outputs = np.empty([beam_search_k, 1]).tolist()
	# First decode
	#DO: [BS x Vocab], dh: [BS x H]
	decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, enc_hidden)
	decoder_output = F.log_softmax(decoder_output, dim=1)
	#decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, enc_hidden)
	# Choose the k options with the highest probability
	topv, topi = decoder_output.topk(beam_search_k)
	for k in range(beam_search_k):
		# Record hidden layer, type: [torch.tensor()]
		hidden_log[k] = decoder_hidden
		# Record probability (sorted in descending order by default), type: [float]
		prob_log[k] += topv.squeeze()[k].item()
		# Log output (corresponding to the probability of prob_log), type: [int]
		decoder_outputs[k].append(topi.squeeze()[k].item())
		decoder_outputs[k].pop(0) # Delete the elements stored during initialization
	# beam search
	for ei in range(max_length - 1):
		# Used to temporarily store the probability for later comparison
		temp_prob_log = torch.tensor([]).to(device)
		temp_output_log = torch.tensor([], dtype=torch.long).to(device)
		temp_hidden_log = []
		for k in range(beam_search_k):
			decoder_input = torch.tensor([decoder_outputs[k][-1]], dtype=torch.long).to(device)
			if decoder_input.item() != eos_token:
				decoder_hidden = hidden_log[k]
				decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, enc_hidden)
				# Preliminary comparison
				topv, topi = decoder_output.topk(beam_search_k)
				topv += prob_log[k]
				temp_prob_log = torch.cat([temp_prob_log, topv], dim=1)
				temp_hidden_log.append(decoder_hidden)
				temp_output_log = torch.cat([temp_output_log, topi], dim=1)
			else: # If it has reached <EOS>
				topv = torch.ones(1, beam_search_k).to(device) * prob_log[k]
				topi = torch.ones(1, beam_search_k).to(device) * eos_token
				# Index should be long
				topi = topi.long()
				temp_prob_log = torch.cat([temp_prob_log, topv], dim=1)
				temp_hidden_log.append(None)
				temp_output_log = torch.cat([temp_output_log, topi], dim=1)
		# Final comparison (the k options with the highest probability are selected from the k*k candidates)
		temp_topv, temp_topi = temp_prob_log.topk(beam_search_k)
		temp_decoder_outputs = decoder_outputs.copy()
		# Record the results (keep the probability in descending order)
		for k in range(beam_search_k):
			ith = int(temp_topi.squeeze()[k].item() / beam_search_k)
			hidden_log[k] = temp_hidden_log[ith]
			prob_log[k] = temp_topv.squeeze()[k].item()
			decoder_outputs[k] = temp_decoder_outputs[ith].copy()
			if temp_output_log.squeeze()[temp_topi.squeeze()[k].item()].item() != eos_token \
					and decoder_outputs[k][-1] != eos_token:
				decoder_outputs[k].append(temp_output_log.squeeze()[temp_topi.squeeze()[k].item()].item())
	# Optimistic return
	return decoder_outputs[0]

#translate_sentence(src, SRC, model, device)
def translate_sentence(src, src_field, model, device, max_len = 30):
	model.eval()
	'''
	tokens = [token.lower() for token in sentence]
	tokens = [src_field.init_token] + tokens + [src_field.eos_token]
	src_indexes = [src_field.vocab.stoi[token] for token in tokens]
	src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
	src_len = torch.LongTensor([len(src_indexes)]).to(device)
	'''
	src_tensor, src_len = translateData(src, src_field)
	with torch.no_grad():
		#forward(self, src, src_len, embedding)
		syntax, content = model.encoder(src_tensor, src_len)
	enc_hidden = torch.cat((syntax, content), dim = 1)
	hidden = enc_hidden
	trg_indexes = [src_field.init_token]
	#attentions = torch.zeros(max_len, 1, ing_len.item()).to(device)
	for i in range(max_len):
		trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
		with torch.no_grad():
			#(self, input, hidden, enc_hidden, embedding)
			output, hidden = model.decoder(trg_tensor, hidden, enc_hidden)
		#attentions[i] = attention
		pred_token = output.argmax(1).item()
		trg_indexes.append(pred_token)
		if pred_token == src_field.eos_token:
			break
	#trg_tokens = sp_gec.decode(trg_indexes)
	#trg_tokens = [src_field.vocab.itos[i] for i in trg_indexes]
	return trg_indexes#, attentions[:len(trg_tokens)-1]

#HERE
def testOP(data, index, tp, decodeType):
	x = open("eval.txt","a")
	x.write(tp+" Set: Index-"+str(index)+"\n")
	#HERE
	x.write("Decoding Type: "+decodeType+"\n")
	trg = vars(data.examples[index])['correction1']
	src = vars(data.examples[index])['orig']
	#HERE
	if decodeType == "greedy":
		translation = translate_sentence(src, SRC, model, device)
	else:
		#bs = Beam_Search(model)
		#forward(self, src, src_field, device, max_len = 30)
		#translation = bs(src, SRC, device)
		translation = beam_search(src, SRC, model, device)
	translation = sp_gec.decode(translation)
	src = sp_gec.decode(src)
	trg = sp_gec.decode(trg)
	print(f'src = {src}')
	print(f'trg = {trg}')
	print(f'predicted trg = {translation}')
	x.write("src = "+src+"\n")
	x.write("Ground trg = "+trg+"\n")
	x.write("Predicted trg = "+translation+"\n")
	x.close()
	#display_attention(src, translation, attention, index, tp)

#HERE
def calculate_bleu(data, src_field, model, device, decodeType, max_len = 30):
	cc = SmoothingFunction()
	sentBleu = 0.0
	sentGleu = 0.0
	trgs = []
	pred_trgs = []
	#bs = Beam_Search(model)
	for datum in tqdm(data):
		trg = vars(datum)['correction1']
		src = vars(datum)['orig']
		#translate_sentence(src, src_field, model, device, max_len = 25)
		#HERE
		if decodeType == "greedy":
			pred_trg = translate_sentence(src, src_field, model, device, max_len)
		else:
			#pred_trg = bs(src, src_field, device)
			pred_trg = beam_search(src, src_field, model, device, max_len)
		#cut off <eos> token
		#HERE
		#pred_trg = pred_trg[1:-1]
		#if len(pred_trg) < 2: pred_trg.append(".")
		sentBleu += sentence_bleu([trg], pred_trg, smoothing_function = cc.method3)
		sentGleu += sentence_gleu([trg], pred_trg)
		pred_trgs.append(pred_trg)
		trgs.append([trg])
	sentBleu = sentBleu/len(data)
	sentGleu = sentGleu/len(data)
	corpusBleu = corpus_bleu(trgs, pred_trgs, smoothing_function = cc.method3)
	corpusGleu = corpus_gleu(trgs, pred_trgs)
	return sentBleu, sentGleu, corpusBleu, corpusGleu

#HERE
def prntBLEU(data, tp, decodeType):
	#HERE
	sentBleu, sentGleu, corpusBleu, corpusGleu = calculate_bleu(data, SRC, model, device, decodeType)
	print(f'Sentence BLEU score = {sentBleu*100:.2f}')
	print(f'Corpus BLEU score = {corpusBleu*100:.2f}')
	print(f'Sentence GLEU score = {sentGleu*100:.2f}')
	print(f'Corpus GLEU score = {corpusGleu*100:.2f}')
	x = open("eval.txt","a")
	#HERE
	x.write("Decoding Type: "+decodeType+"\n\n")
	x.write("\n\nSentence BLEU score for the "+tp+"-set is: "+str(f'{sentBleu*100:.2f}'))
	x.write("\n\nCorpus BLEU score for the "+tp+"-set is: "+str(f'{corpusBleu*100:.2f}'))
	x.write("\n\nSentence GLEU score for the "+tp+"-set is: "+str(f'{sentGleu*100:.2f}'))
	x.write("\n\nCorpus GLEU score for the "+tp+"-set is: "+str(f'{corpusGleu*100:.2f}'))
	x.close()

#getResults(model, test_iterator, criterion, bowCrit, tst, mName = 'tut-model-epoch'+str(i)+'.pt')
#HERE
def getResults(model, test_iterator, criterion, tst, mName, decodeType):
	model.load_state_dict(torch.load(mName))
	test_loss = evaluate(model, test_iterator, criterion)
	x = open("eval.txt","a")
	e = mName.split("epoch")[1].split(".")[0]
	x.write("\n\nEpoch Result: "+e+"\n")
	print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
	x.write(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
	x.close()
	#Random 10 samples.
	#example_idx = [10,20,50,100,200,1000,500,150,300,400]
	#example_idx = len(test_iterator)
	#example_idx = random.sample(range(example_idx), 20)
	#HERE
	#example_idx = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000]
	example_idx = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900]
	#trn, tst, vld
	for i in example_idx:
		#HERE
		testOP(tst, i, "Test", decodeType)
	#Printing BLEU score:
	#HERE
	prntBLEU(tst, "Test", decodeType)
	#prntBLEU(vld, "Validation")
	#prntBLEU(trn, "Train")



def init_weights2(m):
	for name, param in m.named_parameters():
		if 'weight' in name:
			nn.init.normal_(param.data, mean=0, std=0.01)
		else:
			nn.init.constant_(param.data, 0)

def init_weights(m):
	if hasattr(m, 'weight') and m.weight.dim() > 1:
		nn.init.xavier_uniform_(m.weight.data)


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calcReconLoss(output, trg, criterion):
	output_dim = output.shape[-1]
	output = output[1:].view(-1, output_dim)
	trg = trg[1:].view(-1)
	loss = criterion(output, trg)
	return loss

def calcBOWLoss(bow, trg, bowCrit):
	N = len(noSW.vocab)
	trg = trg.transpose(0, 1)#TxB -> BxT
	OH = F.one_hot(trg, N)
	OH = OH.sum(dim = 1).float()
	OH = OH.bool().float()
	loss = bowCrit(bow, OH)
	return loss


#train(model, train_iterator, optimizer, criterion, bowCrit, CLIP)
def train(model, iterator, optimizer, criterion, clip):
	model.train()
	epoch_loss = 0.0
	reconLoss = 0.0
	#simLoss = 0.0
	#bowLoss = 0.0
	batchStartTime = time.time()
	for i, batch in enumerate(iterator):
		if i%100 == 0:
			x = open("training.txt", "a")
			batchEndTime = time.time()
			epoch_mins, epoch_secs = epoch_time(batchStartTime, batchEndTime)
			eLoss = float(epoch_loss/(i+1))
			rLoss = float(reconLoss/(i+1))
			#sLoss = float(simLoss/(i+1))
			#bLoss = float(bowLoss/(i+1))
			print(f'Batch: {i+1:02} | Time: {epoch_mins}m {epoch_secs}s | EpochLoss: {eLoss:.4f} | ReconLoss: {rLoss:.4f}')
			x.write(f'Batch: {i+1:02} | Time: {epoch_mins}m {epoch_secs}s | EpochLoss: {eLoss:.4f} | ReconLoss: {rLoss:.4f}\n')
			x.close()
			batchStartTime = time.time()
			#tv_datafields = [("orig", SRC), ("correction1", SRC), ("correction2", noSW)]
		trg, trg_len = batch.correction1
		src, src_len = batch.orig
		#trgnoSW = batch.correction2
		optimizer.zero_grad()
		#(self, src, src_len, trg, trg_len, teacher_forcing_ratio = 0.50)
		#outputS, contentS, bowS
		outputS = model(src, trg)
		#print()
		#trg = [trg len, batch size]
		#output = [trg len, batch size, output dim]
		#trg = [(trg len - 1) * batch size]
		#output = [(trg len - 1) * batch size, output dim]
		recon1 = calcReconLoss(outputS, trg, criterion)
		#recon2 = calcReconLoss(outputT, trg, criterion)
		loss1 = recon1
		#loss2 = simCrit(contentS, contentT)
		#bow1 = calcBOWLoss(bowS, trgnoSW, bowCrit)
		#bow2 = calcBOWLoss(bowT, trg, bowCrit)
		#loss3 = bow1
		loss = loss1
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		optimizer.step()
		epoch_loss += loss.item()
		reconLoss += loss1.item()
		#simLoss += loss2.item()
		#bowLoss += loss3.item()
	return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
	model.eval()
	epoch_loss = 0
	with torch.no_grad():
		for i, batch in enumerate(iterator):
			#tv_datafields = [("type", None), ("orig", SRC), ("correction", SRC)]
			trg, trg_len = batch.correction1
			src, src_len = batch.orig
			#outputS, contentS, bowS = model(src, src_len, trg, trg_len)
			outputS = model(src, src_len, trg, trg_len, 0.0)
			#output = model(src, src_len, trg, 0) #turn off teacher forcing
			#trg = [trg len, batch size]
			#output = [trg len, batch size, output dim]
			recon1 = calcReconLoss(outputS, trg, criterion)
			#recon2 = calcReconLoss(outputT, trg, criterion)
			loss = recon1
			epoch_loss += loss.item()
	return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs

if __name__ == '__main__':
	mType = sys.argv[1]#train/test
	SEED = 1234
	BATCH_SIZE = 30
	CL_DROPOUT = 0.2
	N_EPOCHS = 51
	CLIP = 1
	LR = 3e-4
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True
	tokenize = lambda x: x.split()
	# sp_gec = load_sp_model("GCEBPE30k.model")
	pkl_file = open('BPE/data.pkl', 'rb')
	data1 = pickle.load(pkl_file)
	sp_gec = data1["sp_gec_orig"]
	#SRC = Field(tokenize = tokenize, init_token='<sos>', eos_token='<eos>', lower = False, include_lengths=True)
	SRC = Field(use_vocab = False, tokenize = sp_gec.encode, init_token = sp_gec.bos_id(), eos_token = sp_gec.eos_id(), pad_token = sp_gec.pad_id(), include_lengths = True)
	#noSW = Field(use_vocab = True, tokenize = tokenize)
	#tv_datafields = [("type", None), ("orig", SRC), ("correction", SRC)]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	tv_datafields = [("orig", SRC), ("correction1", SRC), ("correction2", None)]
	#trn, tst = TabularDataset.splits(path = "../Data", train = 'train100.tsv', test = "testSS.tsv", format='tsv', skip_header=False, fields = tv_datafields)
	trn, tst = TabularDataset.splits(path = ".data/", train = "train300k.tsv", test = "test10k.tsv", format='tsv', skip_header=False, fields = tv_datafields)
	#noSW.build_vocab(trn, tst, min_freq = 1)
	#SRC.build_vocab(trn, tst, min_freq = 2, max_size = 100000, vectors="glove.6B.200d")
	#train_iterator, test_iterator = BucketIterator.splits((trn, tst), batch_size = BATCH_SIZE, sort_within_batch = True, sort_key = lambda x : len(x.orig), device = device)
	train_iterator, test_iterator = BucketIterator.splits((trn, tst), batch_size = BATCH_SIZE, sort_within_batch = True, sort_key = lambda x : len(x.orig), device = device)
	INPUT_DIM = OUTPUT_DIM = len(sp_gec)
	#clVocab = len(noSW.vocab)
	#OUTPUT_DIM = len(TRG.vocab)
	#SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
	SRC_PAD_IDX = sp_gec.pad_id()
	#attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
	#(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout)

	#(self, inDim, vocab, dropout)
	#classif = classifierNet(int(ENC_HID_DIM/2), clVocab, CL_DROPOUT)
	#self, encoder, decoder, classifier, src_pad_idx, input_dim, emb_dim, device
	model = Seq2Seq(SRC_PAD_IDX, INPUT_DIM, device).to(device)
	best_valid_loss = float('inf')
	model.apply(init_weights)
	optimizer = optim.Adam(model.parameters(), lr = LR)
	TRG_PAD_IDX = SRC_PAD_IDX
	criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
	#simCrit = nn.MSELoss()
	#bowCrit = nn.BCEWithLogitsLoss()
	x = open("result.txt","a")
	print(f'The model has {count_parameters(model):,} trainable parameters')
	x.write(f'The model has {count_parameters(model):,} trainable parameters\n')
	x.close()
	if mType == "train":
		for epoch in range(N_EPOCHS):
			y = open("training.txt", "a")
			y.write(f'Epoch: {epoch+1:02}\n')
			y.close()
			x = open("result.txt","a")
			start_time = time.time()
			train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
			valid_loss = evaluate(model, test_iterator, criterion)
			end_time = time.time()
			epoch_mins, epoch_secs = epoch_time(start_time, end_time)
			if valid_loss < best_valid_loss:
				best_valid_loss = valid_loss
				torch.save(model.state_dict(), 'tut-model-epoch'+str(epoch)+'.pt')
			elif (epoch % 5 == 0):
				torch.save(model.state_dict(), 'tut-model-epoch'+str(epoch)+'.pt')
			print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
			x.write(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
			print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
			x.write(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n')
			print(f'\t Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')
			x.write(f'\t Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}\n')
			x.close()
		#Printing/Calculating Results:
	elif mType == "test":
		#HERE
		decodeType = sys.argv[3]
		ep = [int(i) for i in sys.argv[2].split(",")]
		for i in ep:
			#HERE
			getResults(model, test_iterator, criterion, tst, 'tut-model-epoch'+str(i)+'.pt', decodeType)
	elif mType == "continue":
		nE = sys.argv[2]
		N_EPOCHS = sys.argv[3]
		N_EPOCHS = int(N_EPOCHS)
		mName = 'tut-model-epoch'+str(nE)+'.pt'
		model.load_state_dict(torch.load(mName))
		for epoch in range(int(nE)+1, N_EPOCHS):
			y = open("training.txt", "a")
			y.write(f'Epoch: {epoch+1:02}\n')
			y.close()
			x = open("result.txt","a")
			start_time = time.time()
			train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
			valid_loss = evaluate(model, test_iterator, criterion)
			end_time = time.time()
			epoch_mins, epoch_secs = epoch_time(start_time, end_time)
			if valid_loss < best_valid_loss:
				best_valid_loss = valid_loss
				torch.save(model.state_dict(), 'tut-model-epoch'+str(epoch)+'.pt')
			elif (epoch % 5 == 0):
				torch.save(model.state_dict(), 'tut-model-epoch'+str(epoch)+'.pt')
			print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
			x.write(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n')
			print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
			x.write(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n')
			print(f'\t Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')
			x.write(f'\t Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}\n')
			x.close()