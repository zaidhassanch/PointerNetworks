
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

# data = train_data[0:3]
# for example in data:
#     src = example.src
#     trg = example.trg
#     print(">>>> ", src)
#     print(spe_dec.decode(src))
#     print("     ", trg)
#     print(spe_dec.decode(trg))


src_vocab_size = len(spe_dec)
trg_vocab_size = len(spe_dec)
print("src vocabulary size: ", src_vocab_size)
print("trg vocabulary size: ", trg_vocab_size)
embedding_size = 256
src_pad_idx = spe_dec.pad_id()
print("pad_index = ", src_pad_idx)
print("===============================after loading")

model = Transformer(device, embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx).to(device)

load_model = True
save_model = True
learning_rate = 3e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

train(model, device, load_model, save_model, spe_dec, spe_dec, train_data, valid_data, test_data, batch_size)
# running on entire test data takes a while


