import os

import spacy
import pickle
from torchtext.data import Field
from torchtext.datasets import Multi30k, TranslationDataset
from torchtext.data.functional import load_sp_model
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

    #sp_gec = load_sp_model("BPE/GCEBPE30k.model")
    sp_gec = load_sp_model("BPE/zaid_sp_model.model")
    print("print(len(sp_gec)) 1", len(sp_gec))

    bpe_field = Field(use_vocab = False, tokenize = sp_gec.encode, 
	    init_token = sp_gec.bos_id(), eos_token = sp_gec.eos_id(), pad_token = sp_gec.pad_id(), batch_first = True)

    tv_datafields = [("src", bpe_field), ("trg", bpe_field)]
    # german = Field(tokenize=tokenize_ger, lower=True,
    #                init_token="<sos>", eos_token="<eos>",  pad_token="<pad>", unk_token="<unk>")

    # english = Field(
    #     tokenize=tokenize_eng, lower=True,
    #     init_token="<sos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")

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
        exts=(".ennsw", ".en"), fields=tv_datafields,
        # root='.data',
        train='train',
        validation='val',
        test='test2016',
        path = '.data/multi30k'
    )
    print(train_data)

    # example_id = 0
    # ex = train_data.examples[example_id].src
    # dec_ex = sp_gec.decode(ex)
    # print(ex)
    # print(dec_ex)

    # ex = train_data.examples[example_id].trg
    # dec_ex = sp_gec.decode(ex)
    # print(ex)
    # print(dec_ex)

    #====================================================================================
    # exit()

    # train_data, valid_data, test_data = Multi30k.splits(
    #     exts=(".con", ".tgt"), fields=(german, english),
    #     # root='.data',
    #     train='shortouttest300k',
    #     validation='shortout10k',
    #     test='shortout10k',
    #     path='/data/chaudhryz/ank_data'
    # )

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
    # german.build_vocab(train_data, max_size=10000, min_freq=2)
    # english.build_vocab(train_data, max_size=10000, min_freq=2)

    # german.vocab.init_token = "<sos>"
    # german.vocab.eos_token = "<eos>"

    # english.vocab.init_token = "<sos>"
    # english.vocab.eos_token = "<eos>"

    # init_token = sp_gec.bos_id(), eos_token = sp_gec.eos_id()

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

    return sp_gec, train_data, valid_data, test_data
