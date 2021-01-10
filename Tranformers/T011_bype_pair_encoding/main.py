
from utils import translate_sentence, load_checkpoint
import torch
from data_loader import getData #, getData_newMethod
from train import train
from transfomer import Transformer

#german_vocab, english_vocab, train_data, valid_data, test_data = getData_newMethod()
print("===============================before loading")
spe_dec, train_data, valid_data, test_data = getData()
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
    print(">>>> ", src)
    print(spe_dec.decode(src))
    print("     ", trg)
    print(spe_dec.decode(trg))
# exit()

src_vocab_size = len(spe_dec)
trg_vocab_size = len(spe_dec)
print("src vocabulary size: ", src_vocab_size)
print("trg vocabulary size: ", trg_vocab_size)
embedding_size = 512
src_pad_idx = spe_dec.pad_id()#english_vocab.stoi["<pad>"]
print("pad_index = ", src_pad_idx)
print("pad = ", spe_dec.decode(src_pad_idx))
print("===============================after loading")
# exit()

model = Transformer(device, embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx).to(device)

load_model = False
save_model = True
learning_rate = 3e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("RSWI_checkpoint.pth.tar"), model, optimizer)

# sentence = "ein pferd geht unter einer brücke neben einem boot."
#
# translated_sentence = translate_sentence(
#     model, sentence, german, english, device, max_length=50
# )
# sentence = 'The study questions are carefully worded and chosen.'
# sentence = 'a little girl climbing into a wooden playhouse.'

sentence = "man stuffed smiling lion here"

#sentence1 = ['ein', 'pferd', 'geht', 'unter', 'einer', 'brücke', 'neben', 'einem', 'boot', '.']
# sentence1 = ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']
translated_sentence = translate_sentence(model, sentence, spe_dec, spe_dec, device, max_length=100)
# exit()
# print(f"Translated1 example sentence: \n {sentence}")
# print(f"Translated1 example sentence: \n {translated_sentence}")

# exit()
print("===============================going for training ")

train(model, device, load_model, save_model, spe_dec, spe_dec, train_data, valid_data, test_data, batch_size)
# running on entire test data takes a while


