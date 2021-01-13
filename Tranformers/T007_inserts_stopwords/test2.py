# coding=utf-8

from utils import translate_sentence, load_checkpoint_zaid, save_checkpoint_zaid, bleu
import torch
from data import getData, getData_newMethod
from train import train
from transfomer import Transformer

#german_vocab, english_vocab, train_data, valid_data, test_data = getData_newMethod()
print("===============================before loading")
german_vocab, english_vocab, train_data, valid_data, test_data = getData()
print("train_data ", len(train_data.examples))
print("valid_data ", len(valid_data.examples))
print("test_data ", len(test_data.examples))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
batch_size = 32

data = train_data[0:3]

for example in data:
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
print(src_pad_idx)
print(english_vocab.itos[src_pad_idx])
print("===============================after loading ")

model = Transformer(device, embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx).to(device)

load_model = True
save_model = True
learning_rate = 3e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint_zaid("my_checkpoint.pth.tar", model, optimizer)


print("here1")
score = bleu(train_data[1:10], model, german_vocab, english_vocab, device)
print(f"Train Bleu score {score * 100:.2f}")

print("here2")
score = bleu(test_data[1:50], model, german_vocab, english_vocab, device)
print(f"Test Bleu score {score * 100:.2f}")

if save_model:
    save_checkpoint_zaid("my_checkpointx.pth.tar", model, optimizer)

