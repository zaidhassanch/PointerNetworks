import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
	#def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout)
	def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
		super().__init__()
		self.embedding = nn.Embedding(input_dim, emb_dim)
		#HERE
		self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True, num_layers = 2)
		#HERE
		self.fc = nn.Linear(enc_hid_dim * 4, dec_hid_dim)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()
	def forward(self, src, src_len):
		#src = [src len, batch size]
		#src_len = [src len]
		embedded = self.dropout(self.embedding(src))
		#embedded = [src len, batch size, emb dim]
		#HERE
		packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu())
		_, hidden = self.rnn(packed_embedded)
		#packed_outputs is a packed sequence containing all hidden states
		#hidden is now from the final non-padded element in the batch
		#outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
		#outputs is now a non-packed sequence, all hidden states obtained
		#when the input is a pad token are all zeros
		#outputs = [src len, batch size, hid dim * num directions]
		#hidden = [n layers * num directions, batch size, hid dim]
		#hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
		#outputs are always from the last layer
		#hidden [-2, :, : ] is the last of the forwards RNN
		#hidden [-1, :, : ] is the last of the backwards RNN
		#initial decoder hidden is final hidden state of the forwards and backwards
		#encoder RNNs fed through a linear layer
		#HERE
		p = hidden[0,:,:]
		for i in range(1,hidden.shape[0]):
			p = torch.cat((hidden[i,:,:], p), dim = 1)
		hf = self.fc(p)
		hidden = self.relu(hf)
		#outputs = [src len, batch size, enc hid dim * 2]
		#hidden = [batch size, dec hid dim]
		syntax, content = torch.chunk(hidden, chunks = 2, dim = 1)
		return syntax, content

class Decoder(nn.Module):
	def __init__(self, output_dim, emb_dim, dec_hid_dim, dropout):
		super().__init__()
		self.output_dim = output_dim
		#self.attention = attention
		self.embedding = nn.Embedding(output_dim, emb_dim)
		#self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
		self.rnn = nn.GRU(emb_dim + dec_hid_dim, dec_hid_dim)
		self.fc_out = nn.Linear(dec_hid_dim, output_dim)
		self.dropout = nn.Dropout(dropout)
	def forward(self, input, hidden, enc_hidden):
		#input = [batch size]
		#hidden = [batch size, dec hid dim]
		#encoder_outputs = [src len, batch size, enc hid dim * 2]
		#mask = [batch size, src len]
		input = input.unsqueeze(0)
		#input = [1, batch size]
		embedded = self.dropout(self.embedding(input))
		#embedded = [1, batch size, emb dim]
		#a = self.attention(hidden, encoder_outputs, mask)
		#a = [batch size, src len]
		#a = a.unsqueeze(1)
		#a = [batch size, 1, src len]
		#encoder_outputs = encoder_outputs.permute(1, 0, 2)
		#encoder_outputs = [batch size, src len, enc hid dim * 2]
		#weighted = torch.bmm(a, encoder_outputs)
		#weighted = [batch size, 1, enc hid dim * 2]
		#weighted = weighted.permute(1, 0, 2)
		#weighted = [1, batch size, enc hid dim * 2]
		#rnn_input = torch.cat((embedded, weighted), dim = 2)
		#rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
		rnn_input = torch.cat((embedded, enc_hidden.unsqueeze(0)), dim = 2)
		output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
		#output = [seq len, batch size, dec hid dim * n directions]
		#hidden = [n layers * n directions, batch size, dec hid dim]
		#seq len, n layers and n directions will always be 1 in this decoder, therefore:
		#output = [1, batch size, dec hid dim]
		#hidden = [1, batch size, dec hid dim]
		#this also means that output == hidden
		assert (output == hidden).all()
		#embedded = embedded.squeeze(0)
		output = output.squeeze(0)
		#weighted = weighted.squeeze(0)
		prediction = self.fc_out(output)
		#prediction = self.fc_out(torch.cat((output, embedded), dim = 1))
		#prediction = [batch size, output dim]
		return prediction, hidden.squeeze(0)

class classifierNet(nn.Module):
	def __init__(self, inDim, vocab, dropout):
		super().__init__()
		#self.inDim = inDim
		self.vocab = vocab
		#self.relu = nn.ReLU()
		self.net = nn.Sequential(
			nn.Linear(inDim, inDim*2),
			nn.ReLU(),
			nn.Linear(inDim*2, inDim*4),
			nn.Dropout(dropout),
			nn.Linear(inDim*4, inDim*8),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(inDim*8, vocab)
			)
	def forward(self, input):
		x = self.net(input)
		return x

class Seq2Seq(nn.Module):
	def __init__(self, src_pad_idx, input_dim, device):
		super().__init__()
		ENC_EMB_DIM = 300
		DEC_EMB_DIM = 300
		ENC_HID_DIM = 512
		DEC_HID_DIM = 512
		ENC_DROPOUT = 0.2
		DEC_DROPOUT = 0.2
		CL_DROPOUT = 0.2
		self.encoder = Encoder(input_dim, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
		# (self, output_dim, emb_dim, dec_hid_dim, dropout)
		self.decoder = Decoder(input_dim, DEC_EMB_DIM, DEC_HID_DIM, DEC_DROPOUT)

		#self.classifier = classifier
		self.src_pad_idx = src_pad_idx
		self.device = device
		self.input_dim = input_dim


		#self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx = self.src_pad_idx)
	def genOP(self, src, src_len, trg, teacher_forcing_ratio = 0.50):
		'''
		trg_lenD = trg.shape[0]
		batch_size = src.shape[1]
		#batch_size = src.shape[1]
		#trg_len = trg.shape[0]
		trg_vocab_size = self.input_dim
		#tensor to store decoder outputs
		outputsS = torch.zeros(trg_lenD, batch_size, trg_vocab_size).to(self.device)
		'''
		trg_len = trg.shape[0]
		batch_size = src.shape[1]
		#batch_size = src.shape[1]
		#trg_len = trg.shape[0]
		trg_vocab_size = self.input_dim
		outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
		#self, src, src_len, embedding
		syntax, content = self.encoder(src, src_len)
		enc_hidden = torch.cat((syntax, content), dim = 1)
		hidden = enc_hidden
		input = trg[0,:]
		#mask = [batch size, src len]
		for t in range(1, trg_len):
			#self, input, hidden, enc_hidden, embedding
			output, hidden = self.decoder(input, hidden, enc_hidden)
			outputs[t] = output
			#decide if we are going to use teacher forcing or not
			teacher_force = random.random() < teacher_forcing_ratio
			#get the highest predicted token from our predictions
			top1 = output.argmax(1)
			#if teacher forcing, use actual next token as next input
			#if not, use predicted token
			input = trg[t] if teacher_force else top1
		return outputs
	def forward(self, src, src_len, trg, trg_len, teacher_forcing_ratio = 0.50):
		#src = [src len, batch size]
		#trgnoSW = []
		#src_len = [batch size]
		#trg = [trg len, batch size]
		#teacher_forcing_ratio is probability to use teacher forcing
		#e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
		#genOP(self, src, src_len, trg, teacher_forcing_ratio = 0.50
		outputS = self.genOP(src, src_len, trg, teacher_forcing_ratio)
		#contentT, outputT = self.genOP(trg, trg_len, trg, teacher_forcing_ratio)
		#bowS = self.classifier(contentS)
		#bowT = self.classifier(contentT)
		#return outputS, outputT, contentS, contentT, bowS, bowT
		return outputS