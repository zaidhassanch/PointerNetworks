import torch
import random
from torch import nn

class Encoder(nn.Module):
	def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
		super().__init__()
		self.embedding = nn.Embedding(input_dim, emb_dim)
		self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True, num_layers = 2)
		self.fc = nn.Linear(enc_hid_dim * 4, dec_hid_dim)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()

	def forward(self, src):                 # (13, 30)
		embedded1 = self.embedding(src)     # (13, 30, 300)
		embedded = self.dropout(embedded1)  # (13, 30, 300)
		_, hidden = self.rnn(embedded)      # _ (13, 30, 1024), hidden (4, 30, 512)

		p = hidden[0,:,:]                                # p (30, 512)
		for i in range(1,hidden.shape[0]):
			p = torch.cat((hidden[i,:,:], p), dim = 1)

		# p (30, 2048)
		hf = self.fc(p)                                  # hf (30, 512)
		hidden = self.relu(hf)                           # hidden (30, 512)
		syntax, content = torch.chunk(hidden, chunks = 2, dim = 1) # (30, 256) each
		return syntax, content


class Decoder(nn.Module):
	def __init__(self, output_dim, emb_dim, dec_hid_dim, dropout):
		super().__init__()
		self.output_dim = output_dim
		self.embedding = nn.Embedding(output_dim, emb_dim)
		self.rnn = nn.GRU(emb_dim + dec_hid_dim, dec_hid_dim)
		self.fc_out = nn.Linear(dec_hid_dim, output_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, target, hidden, enc_hidden):
		target = target.unsqueeze(0)                                            # (1, 30)
		embedded1 = self.embedding(target)                                     # (1, 30, 300)
		embedded = self.dropout(embedded1)                                    # (1, 30, 300)
		rnn_input = torch.cat((embedded, enc_hidden), dim = 2)   # (1, 30, 812) = (1,30,300)+(1,30,512)
		output, hidden = self.rnn(rnn_input, hidden)             # (1, 30, 512), (1,30,512)
		assert (output == hidden).all()
		output = output.squeeze(0)                                            # (30, 512)
		prediction = self.fc_out(output)                                      # (30, 30000)
		return prediction, hidden                                             # (30, 30000), (30, 512)

class classifierNet(nn.Module):
	def __init__(self, inDim, vocab, dropout):
		super().__init__()
		self.vocab = vocab
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
		ENC_EMB_DIM = 300;	DEC_EMB_DIM = 300
		ENC_HID_DIM = 512;  DEC_HID_DIM = 512
		ENC_DROPOUT = 0.2;  DEC_DROPOUT = 0.2

		self.encoder = Encoder(input_dim, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
		self.decoder = Decoder(input_dim, DEC_EMB_DIM, DEC_HID_DIM, DEC_DROPOUT)
		self.src_pad_idx = src_pad_idx
		self.device = device
		self.input_dim = input_dim

	def forward(self, src, trg, teacher_forcing_ratio = 0.50):

		trg_len = trg.shape[0]
		batch_size = src.shape[1]
		trg_vocab_size = self.input_dim
		outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
		syntax, content = self.encoder(src)
		enc_hidden = torch.cat((syntax, content), dim = 1)
		enc_hidden = enc_hidden.unsqueeze(0)
		hidden = enc_hidden
		target = trg[0,:]

		for t in range(1, trg_len):
			output, hidden = self.decoder(target, hidden, enc_hidden)
			outputs[t] = output
			teacher_force = random.random() < teacher_forcing_ratio
			top1 = output.argmax(1)
			target = trg[t] if teacher_force else top1
		return outputs
