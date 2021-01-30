
from utils import translate_sentence, load_checkpoint
import torch
from data import getData
from train import train
from transfomer import Transformer
from seq2seq import Seq2Seq

LOAD_NEW_METHOD = True
batch_size = 32
print("===============================before loading")
# german_vocab, english_vocab, train_data, valid_data, test_data = getData_newMethod(LOAD_NEW_METHOD)
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
print(src_pad_idx)
print(english_vocab.itos[src_pad_idx])
print("===============================after loading ")

#model = Transformer(device, embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx).to(device)
model = Seq2Seq(src_pad_idx, src_vocab_size, trg_vocab_size, device).to(device)

load_model = False
save_model = True
learning_rate = 3e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "ein pferd geht unter einer brücke neben einem boot."
#
# translated_sentence = translate_sentence(
#     model, sentence, german, english, device, max_length=50
# )
# sentence = 'The study questions are carefully worded and chosen.'
# sentence = 'a little girl climbing into a wooden playhouse.'

# sentence = "is man lion a stuffed A at smiling."

#sentence1 = ['ein', 'pferd', 'geht', 'unter', 'einer', 'brücke', 'neben', 'einem', 'boot', '.']
# sentence1 = ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']
translated_sentence = translate_sentence(model, sentence, german_vocab, english_vocab, device, max_length=50)
# exit()
# print(f"Translated1 example sentence: \n {sentence}")
# print(f"Translated1 example sentence: \n {translated_sentence}")

# exit()
print("===============================going for training ")

train(model, device, load_model, save_model, 
	german_vocab, english_vocab, train_data, valid_data, test_data, batch_size, LOAD_NEW_METHOD)
# running on entire test data takes a while


