import os
import pickle

import spacy
from torchtext.data import Field
from torchtext.datasets import Multi30k, TranslationDataset
from dataloader import getData2
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

spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def getData(LOAD_NEW_METHOD, USE_BPE):
    return getData2(config.g_tok, config.e_tok, config.USE_BPE)

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
