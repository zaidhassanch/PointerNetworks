
from utils import translate_sentence, load_checkpoint
import torch
from data import getData
from train import train
from transfomer import Transformer

german_vocab, english_vocab, train_data, valid_data, test_data = getData()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

data = train_data[0:3]

for example in data:
    src = example.src
    trg = example.trg
    print(dir(example))
    print(">> ", src)
    print("   ", trg)

# exit()
# Model hyperparameters
src_vocab_size = len(german_vocab)
trg_vocab_size = len(english_vocab)
embedding_size = 512
src_pad_idx = english_vocab.stoi["<pad>"]
print(src_pad_idx)
print(english_vocab.itos[src_pad_idx])

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
sentence1 = ['ein', 'pferd', 'geht', 'unter', 'einer', 'brücke', 'neben', 'einem', 'boot', '.']
sentence1 = ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']
translated_sentence = translate_sentence(model, sentence1, german_vocab, english_vocab, device, max_length=50)
# exit()
# print(f"Translated1 example sentence: \n {sentence}")
# print(f"Translated1 example sentence: \n {translated_sentence}")

# exit()

train(model, device, load_model, save_model, german_vocab, english_vocab, train_data, valid_data, test_data, batch_size)
# running on entire test data takes a while


