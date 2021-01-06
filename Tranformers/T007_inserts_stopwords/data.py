import os

import spacy
import pickle
from torchtext.data import Field
from torchtext.datasets import Multi30k, TranslationDataset
#from atext import getData2
"""
To install spacy languages do:
python -m spacy download en
python -m spacy download de
"""
from torchtext.data.utils import get_tokenizer

spacy_ger = spacy.load("en")
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
    german = Field(tokenize=tokenize_ger, lower=True,
                   init_token="<sos>", eos_token="<eos>",  pad_token="<pad>", unk_token="<unk>")

    english = Field(
        tokenize=tokenize_eng, lower=True,
        init_token="<sos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")

    print("===============================before ")
    # train_data, valid_data, test_data = Multi30k.splits(
    #     exts=(".ennsw", ".en"), fields=(german, english),
    #     # root='.data',
    #     train='train',
    #     validation='val',
    #     test='test2016',
    #     path = '.data/multi30k'
    # )

    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".con", ".tgt"), fields=(german, english),
        # root='.data',
        train='out10k',
        validation='out10k',
        test='out10k',
        path='/data/chaudhryz/ank_data'
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


    # build vocabulary

    # why is the vocabulary size the same for both datasets
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



    # store multi30k vocabulary

    # a = {'GermanVocab': german.vocab, 'EnglishVocab': english.vocab}

    # with open('filename.pickle', 'wb') as handle:
    #     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # use multi30k's vocabulary

    # with open('filename.pickle', 'rb') as handle:
    #     b = pickle.load(handle)
    #
    # german.vocab = b['GermanVocab']
    # english.vocab = b['EnglishVocab']

    #
    # print
    # a == b

    return german.vocab, english.vocab, train_data, valid_data, test_data
