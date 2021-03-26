import os
import pickle
import io

import spacy
from torchtext.data import Field
from torchtext.datasets import Multi30k, TranslationDataset

from configs import config
from torchtext.data.functional import generate_sp_model, load_sp_model

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

from configs import config

"""
To install spacy languages do:
python -m spacy download en
python -m spacy download de
"""
from torchtext.data.utils import get_tokenizer

def data_process_bpe(ger_path, eng_path, ger_voc, eng_voc, ger_tok, eng_tok):
  raw_de_iter = iter(io.open(ger_path, encoding="utf8"))
  raw_en_iter = iter(io.open(eng_path, encoding="utf8"))
  data = []
  for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
    raw_en = raw_en.rstrip()
    raw_de = raw_de.rstrip()
    de_tensor_ = torch.tensor(ger_tok.encode(raw_de),
                            dtype=torch.long)

    en_tensor_ = torch.tensor(eng_tok.encode(raw_en),
                            dtype=torch.long)

    # tokens = [token.lower() for token in eng_tok(raw_en)]
    # print(tokens)
    #
    # text_to_indices = [eng_voc[token] for token in tokens]
    # translated_sentence = [eng_voc.itos[idx] for idx in text_to_indices]
    # print(raw_en)
    # print(text_to_indices)
    # print(translated_sentence)
    # exit()
    data.append((de_tensor_, en_tensor_))
  return data

def getData2(ger_tok, eng_tok, USE_BPE):
    ger_voc = ger_tok #vocab from training data
    eng_voc = eng_tok #vocab from training data

    train_data = data_process_bpe(config.TRAIN_SRC, config.TRAIN_TGT, ger_voc, eng_voc, ger_tok, eng_tok)
    # exit()
    val_data = data_process_bpe(config.VAL_SRC, config.VAL_TGT, ger_voc, eng_voc, ger_tok, eng_tok)
    test_data = data_process_bpe(config.TEST_SRC, config.TEST_TGT, ger_voc, eng_voc, ger_tok, eng_tok)

    print("train_data ", len(train_data))
    print("valid_data ", len(val_data))
    print("test_data ", len(test_data))

    return ger_voc, eng_voc, train_data, val_data, test_data


class MyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        # self.german_vocab, self.english_vocab, self.train_data, self.valid_data, \
        #     self.test_data = german_vocab, english_vocab, train_data, valid_data, test_data
        self.german_vocab, self.english_vocab, self.train_data, self.valid_data, \
                                        self.test_data = getData2(config.g_tok, config.e_tok, config.USE_BPE)
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

        self.PAD_IDX = self.german_vocab.pad_id()
        self.SOS_IDX = self.german_vocab.bos_id()
        self.EOS_IDX = self.german_vocab.eos_id()
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

    # def generate_batch(self, data_batch):
    #     de_batch, en_batch = [], []
    #     for (de_item, en_item) in data_batch:
    #         de_batch.append(torch.cat([torch.tensor([self.SOS_IDX]), de_item, torch.tensor([self.EOS_IDX])], dim=0))
    #         en_batch.append(torch.cat([torch.tensor([self.SOS_IDX]), en_item, torch.tensor([self.EOS_IDX])], dim=0))
    #     de_batch = pad_sequence(de_batch, padding_value=self.PAD_IDX)
    #     en_batch = pad_sequence(en_batch, padding_value=self.PAD_IDX)
    #     return de_batch, en_batch

    def generate_batch(self, data_batch):
        PAD_IDX = 1  # german_vocab['<pad>']
        SOS_IDX = 2  # german_vocab['<sos>']
        EOS_IDX = 3  # german_vocab['<eos>']

        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_batch.append(torch.cat([torch.tensor([SOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([SOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
        return de_batch, en_batch
