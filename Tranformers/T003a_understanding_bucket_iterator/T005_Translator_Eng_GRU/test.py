from utils import translate_sentence, load_checkpoint
import torch
from data import getData, getData_new_method
from train import train
from transfomer import Transformer

LOAD_NEW_METHOD = True

german_vocab, english_vocab, train_data, valid_data, test_data = getData_new_method()
print("===============================before loading")
# german_vocab, english_vocab, train_data, valid_data, test_data = getData(LOAD_NEW_METHOD)
# print("train_data ", len(train_data.examples))
# print("valid_data ", len(valid_data.examples))
# print("test_data ", len(test_data.examples))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

data = train_data[0:3]

for example in data:
    src = example
    trg = example
    print(">> ", src)
    print("   ", trg)

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
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# srcFile = open("/data/chaudhryz/uwstudent1/GDATA/test.src", "r")
# tgtFile = open("/data/chaudhryz/uwstudent1/GDATA/test.tgt", "r")

srcFile = open(".data/multi30k/my.en", "r")
tgtFile = open(".data/multi30k/my.en", "r")

for i in range(2):
    srcLine = srcFile.readline()
    srcLine = srcLine.strip()
    tgtLine = tgtFile.readline()
    tgtLine = tgtLine.strip()

    print(i+1, "=", srcLine)
    print(i+1, "=", tgtLine)
    translated_sentence = translate_sentence(model, srcLine, german_vocab, english_vocab, device, max_length=50)
    print(i+1, "===", translated_sentence)
srcFile.close()
tgtFile.close()
# for line in fp:
#     count += 1
#     fw.write(line)
#     if(count == int(args.lines)):
#         print(f">>>>>>> output lines:{args.lines} written successfully to {outfile}")
#         break

# translated_sentence = translate_sentence(model, sentence, german_vocab, english_vocab, device, max_length=50)


