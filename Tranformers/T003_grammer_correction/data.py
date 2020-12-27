import os

import spacy
from torchtext.data import Field
from torchtext.datasets import Multi30k, TranslationDataset
#from atext import getData2
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

def getData_newMethod():
    g_tok = get_tokenizer('spacy', language='de')
    e_tok = get_tokenizer('spacy', language='en')
    return getData2(g_tok, e_tok)


def getData():
    german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

    english = Field(
        tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
    )

    print("===============================before ")
    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".de", ".en"), fields=(german, english),
        # root='.data',
        train='train',
        validation='val',
        test='test2016',
        path = '.data/multi30k'
    )
    #The study’s questions are carefully worded and chosen.
    # The study questions were carefully worded and chosen.

    # train_data, valid_data, test_data = Multi30k.splits(
    #     exts=(".src", ".tgt"), fields=(german, english),
    #     # root='.data',
    #     train='test',
    #     validation='valid',
    #     test='valid',
    #     path = '/data/chaudhryz/uwstudent1/data'
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


    return german.vocab, english.vocab, train_data, valid_data, test_data
