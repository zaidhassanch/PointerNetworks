
from utils import translate_sentence, load_checkpoint #abc
import torch
from data_loader import getData #, getData_newMethod
from train import train
from transfomer import Transformer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", "-path", help="the path")
parser.add_argument("--trainFile", "-trn", help="the train file")
parser.add_argument("--valFile", "-val", help="the validation file")
parser.add_argument("--testFile", "-tst", help="the test file")
args = parser.parse_args()

assert(args.trainFile)
assert(args.valFile)
assert(args.testFile)

#german_vocab, english_vocab, train_data, valid_data, test_data = getData_newMethod()
print("===============================before loading")

[]
spe_dec, train_data, valid_data, test_data = getData(args.path, args.trainFile, 
                                                args.valFile, args.testFile)
print("train_data ", len(train_data.examples))
print("valid_data ", len(valid_data.examples))
print("test_data ", len(test_data.examples))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


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

load_model = False
save_model = True
learning_rate = 3e-4
batch_size = 512
num_batches = len(train_data.examples) / batch_size
train(num_batches, learning_rate, model, device, load_model, save_model, spe_dec, spe_dec, train_data, valid_data, test_data, batch_size)
# running on entire test data takes a while


