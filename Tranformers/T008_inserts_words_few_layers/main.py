
from utils import translate_sentence, load_checkpoint
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
embedding_size = 128
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
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# sentence = "ein pferd geht unter einer brücke neben einem boot."
#
# translated_sentence = translate_sentence(
#     model, sentence, german, english, device, max_length=50
# )
# sentence = 'The study questions are carefully worded and chosen.'
# sentence = 'a little girl climbing into a wooden playhouse.'

sentence = "man stuffed smiling lion"

#sentence1 = ['ein', 'pferd', 'geht', 'unter', 'einer', 'brücke', 'neben', 'einem', 'boot', '.']
# sentence1 = ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']
translated_sentence = translate_sentence(model, sentence, german_vocab, english_vocab, device, max_length=50)
# exit()
# print(f"Translated1 example sentence: \n {sentence}")
# print(f"Translated1 example sentence: \n {translated_sentence}")

# exit()
print("===============================going for training ")

train(model, device, load_model, save_model, german_vocab, english_vocab, train_data, valid_data, test_data, batch_size)
# running on entire test data takes a while


