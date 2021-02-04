
import torch
from seq2seq import  Seq2Seq, attn # 28epochs: PRE:  ['a', 'horse', 'walking', 'beside', 'a', 'boat', 'under', 'a', 'bridge', '.', '<eos>']
								   # 38epochs: PRE:  ['a', 'horse', 'is', 'walking', 'beside', 'a', 'boat', 'under', 'a', 'bridge', '.', '<eos>']
# from seq2seqblvt import  Seq2Seq, attn
from train import train1, train

from data import getData
BATCH_SIZE = 32
LOAD_NEW_METHOD = True

SRC, TRG, train_data, valid_data, test_data = getData(LOAD_NEW_METHOD)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = len(SRC)
OUTPUT_DIM = len(TRG)
SRC_PAD_IDX = SRC.stoi["<pad>"]

src_vocab_size = INPUT_DIM
trg_vocab_size = OUTPUT_DIM
model = Seq2Seq(SRC_PAD_IDX, src_vocab_size, trg_vocab_size, device).to(device)
#model.load_state_dict(torch.load('tut4-model-saved.pt'))
load_model = False
save_model = True
learning_rate = 3e-4
TRG_PAD_IDX = TRG.stoi["<pad>"]



train1(model, device, load_model, save_model,
 	SRC, TRG, train_data, valid_data, test_data, BATCH_SIZE, LOAD_NEW_METHOD, attn)
