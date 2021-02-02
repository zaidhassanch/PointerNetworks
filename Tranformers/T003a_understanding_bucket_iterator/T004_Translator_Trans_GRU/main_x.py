
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import math
import time
from seq2seqblvt import  Seq2Seq
from train_x import train, evaluate, epoch_time
from utils_x import calculate_bleu
from utils_x import translate_sentence

from data import getData

SRC, TRG, train_data, valid_data, test_data = getData(False)

BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
     batch_size = BATCH_SIZE,
     sort_within_batch = True,
     sort_key = lambda x : len(x.src),
     device = device)


INPUT_DIM = len(SRC)
OUTPUT_DIM = len(TRG)
SRC_PAD_IDX = SRC.stoi[SRC.pad_token]

src_vocab_size = INPUT_DIM
trg_vocab_size = OUTPUT_DIM
model = Seq2Seq(SRC_PAD_IDX, src_vocab_size, trg_vocab_size, device).to(device)


optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = TRG.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')


for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    src = "ein pferd geht unter einer brücke neben einem boot ."
    translation, attention = translate_sentence(model, src, SRC, TRG, device)
    print("SRC: ", src)
    print("PRE: ",translation)
