import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import spacy

from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence

import io
import os
import numpy as np
import random
import time
from collections import Counter
from configs import config

USE_BPE = config.USE_BPE

def build_vocab(filepath, tokenizer):
  counter = Counter()
  with io.open(filepath, encoding="utf8") as f:
    for string_ in f:
      string_ = string_.rstrip()
      counter.update(tokenizer(string_.lower()))

  vocab = Vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'], max_size=10000, min_freq=2);
  vocab.init_token = '<sos>'
  vocab.eos_token = '<eos>'
  return vocab

def data_process(ger_path, eng_path, ger_voc, eng_voc, ger_tok, eng_tok):
  raw_de_iter = iter(io.open(ger_path, encoding="utf8"))
  raw_en_iter = iter(io.open(eng_path, encoding="utf8"))
  data = []
  for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
    raw_en = raw_en.lower().rstrip()
    raw_de = raw_de.lower().rstrip()

    de_tensor_ = torch.tensor([ger_voc[token] for token in ger_tok(raw_de)],
                            dtype=torch.long)
    en_tensor_ = torch.tensor([eng_voc[token] for token in eng_tok(raw_en)],
                            dtype=torch.long)

    data.append((de_tensor_, en_tensor_))
  return data
# german.build_vocab(train_data, max_size=10000, min_freq=2)
# english.build_vocab(train_data, max_size=10000, min_freq=2)
def getData2(ger_tok, eng_tok):
    # TODO
    TRAIN_SRC = config.TRAIN_SRC
    TRAIN_TGT = config.TRAIN_TGT
    VAL_SRC = config.VAL_SRC
    VAL_TGT = config.VAL_TGT
    TEST_SRC = config.TEST_SRC
    TEST_TGT = config.TEST_TGT

    ger_voc = build_vocab(TRAIN_SRC, ger_tok) #vocab from training data
    eng_voc = build_vocab(TRAIN_TGT, eng_tok) #vocab from training data

    train_data = data_process(TRAIN_SRC, TRAIN_TGT, ger_voc, eng_voc, ger_tok, eng_tok)
    val_data = data_process(VAL_SRC, VAL_TGT, ger_voc, eng_voc, ger_tok, eng_tok)
    test_data = data_process(TEST_SRC, TEST_TGT, ger_voc, eng_voc, ger_tok, eng_tok)

    print("Input Vocab", len(ger_voc))
    print("Output Vocab", len(eng_voc))
    print("train_data ", len(train_data))
    print("valid_data ", len(val_data))
    print("test_data ", len(test_data))

    return ger_voc, eng_voc, train_data, val_data, test_data

class MyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.german_vocab, self.english_vocab, self.train_data, self.valid_data, \
                                        self.test_data = getData2(config.g_tok, config.e_tok)
    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage_name):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.german_vocab_setup = self.german_vocab
        self.english_vocab_setup = self.english_vocab
        self.train_data_setup = self.train_data
        self.valid_data_setup = self.valid_data
        self.test_data_setup = self.test_data

        self.PAD_IDX = self.german_vocab['<pad>']
        self.SOS_IDX = self.german_vocab['<sos>']
        self.EOS_IDX = self.german_vocab['<eos>']
        print(stage_name)

    def train_dataloader(self):
        train_iter = DataLoader(self.train_data_setup, batch_size=config.BATCH_SIZE,
                                shuffle=True, collate_fn=self.generate_batch)
        return train_iter
    def val_dataloader(self):
        valid_iter = DataLoader(self.valid_data_setup, batch_size=config.BATCH_SIZE,
                                shuffle=True, collate_fn=self.generate_batch)
        return valid_iter
    def test_dataloader(self):
        test_iter = DataLoader(self.test_data_setup, batch_size=config.BATCH_SIZE,
                               shuffle=True, collate_fn=self.generate_batch)
        return test_iter

    def generate_batch(self, data_batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_batch.append(torch.cat([torch.tensor([self.SOS_IDX]), de_item, torch.tensor([self.EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([self.SOS_IDX]), en_item, torch.tensor([self.EOS_IDX])], dim=0))
        de_batch = pad_sequence(de_batch, padding_value=self.PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=self.PAD_IDX)
        return de_batch, en_batch
