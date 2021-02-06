
from utils import load_checkpoint
import torch
from data import getData
from train import train, train1
from transfomer import Transformer
#from seq2seq import  Seq2Seq # 28epochs: PRE:  ['a', 'horse', 'walking', 'beside', 'a', 'boat', 'under', 'a', 'bridge', '.', '<eos>']
								   # 38epochs: PRE:  ['a', 'horse', 'is', 'walking', 'beside', 'a', 'boat', 'under', 'a', 'bridge', '.', '<eos>']

from seq2seq_attn import  Seq2Seq

LOAD_NEW_METHOD = False
batch_size = 32
print("===============================before loading")
german_vocab, english_vocab, train_data, valid_data, test_data = getData(LOAD_NEW_METHOD)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "gpu"
data = train_data[0:3]

for example in data:
    if LOAD_NEW_METHOD:
        src = example[0]
        trg = example[1]
        src = [german_vocab.itos[idx] for idx in src]
        trg = [english_vocab.itos[idx] for idx in trg]
    else:
        src = example.src
        trg = example.trg
    print(">> ", src)
    print("   ", trg)
# exit()

src_vocab_size = len(german_vocab)
trg_vocab_size = len(english_vocab)
print("src vocabulary size: ", src_vocab_size)
print("trg vocabulary size: ", trg_vocab_size)
embedding_size = 512
src_pad_idx = english_vocab.stoi["<pad>"]
TRG_EOS_TOKEN = german_vocab.stoi[german_vocab.eos_token]
print(src_pad_idx)
print(english_vocab.itos[src_pad_idx])
print("===============================after loading ")


#model = Transformer(device, embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx).to(device)
model = Seq2Seq(src_pad_idx, src_vocab_size, trg_vocab_size, device, TRG_EOS_TOKEN).to(device)

load_model = False
save_model = True
learning_rate = 3e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

train1(model, device, load_model, save_model,
	german_vocab, english_vocab, train_data, valid_data, test_data, batch_size, LOAD_NEW_METHOD)
# train(model, device, load_model, save_model,
# 	german_vocab, english_vocab, train_data, valid_data, test_data, batch_size, LOAD_NEW_METHOD)


