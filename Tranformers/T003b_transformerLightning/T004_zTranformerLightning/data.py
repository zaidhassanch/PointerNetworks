import os
import pickle

import spacy
from torchtext.data import Field
from torchtext.datasets import Multi30k, TranslationDataset
from dataloader import getData2
from configs import config
from torchtext.data.functional import generate_sp_model, load_sp_model
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
    if LOAD_NEW_METHOD:
        return getData_new_method(USE_BPE)
    else:
        return getData_old_method(USE_BPE)

def getData_new_method(USE_BPE):
    if USE_BPE==False:
        g_tok = get_tokenizer('spacy', language='de')
        e_tok = get_tokenizer('spacy', language='en')
        return getData2(g_tok, e_tok, USE_BPE)
    else:
        if config.BPE_FROM_PICKLE:
            pkl_file = open('BPE/data.pkl', 'rb')
            data1 = pickle.load(pkl_file)
            sp_gec = data1["sp_gec_orig"]
        else:
            sp_gec = load_sp_model(config.BPE_PATH)
        return getData2(sp_gec, sp_gec, USE_BPE)

def getData_old_method(USE_BPE):

    if USE_BPE == False:
        german = Field(tokenize=tokenize_ger, lower=True,
                       init_token="<sos>", eos_token="<eos>",  pad_token="<pad>", unk_token="<unk>")

        english = Field(
            tokenize=tokenize_eng, lower=True,
            init_token="<sos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")

        # print("===============================before ")
        train_data, valid_data, test_data = Multi30k.splits(
            exts=(".de", ".en"), fields=(german, english),
            # root='.data',
            train='train',
            validation='val',
            test='test2016',
            path = '../../../data/multi30k'
        )
        #The study’s questions are carefully worded and chosen.
        # The study questions were carefully worded and chosen.

        # train_data, valid_data, test_data = Multi30k.splits(
        #     exts=(".src", ".tgt"), fields=(german, english),
        #     # root='.data',
        #     train='train',
        #     validation='valid',
        #     test='test',
        #     path = '/data/chaudhryz/uwstudent1/GDATA'
        # )



        german.build_vocab(train_data, max_size=10000, min_freq=2)
        english.build_vocab(train_data, max_size=10000, min_freq=2)

        german.vocab.init_token = "<sos>"
        german.vocab.eos_token = "<eos>"

        english.vocab.init_token = "<sos>"
        english.vocab.eos_token = "<eos>"
        # print("Train")
        # for i in range(10):
        #     #print(train_data[i].src, train_data[i].trg)
        #     printSent(train_data[i].src)
        #     printSent(train_data[i].trg)

        # print("Test")
        # for i in range(10):
        #     #print(train_data[i].src, train_data[i].trg)
        #     printSent(test_data[i].src)
        #     printSent(test_data[i].trg)
        # exit()


        print("train_data ", len(train_data.examples))
        print("valid_data ", len(valid_data.examples))
        print("test_data ", len(test_data.examples))

        return german.vocab, english.vocab, train_data, valid_data, test_data

    else:
        print("Not Implemented")
        exit()
