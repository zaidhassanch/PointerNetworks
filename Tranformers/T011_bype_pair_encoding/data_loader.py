# import os

# import spacy
# import pickle
from torchtext.data import Field, TabularDataset
from torchtext.datasets import Multi30k #, TranslationDataset
from torchtext.data.functional import load_sp_model, generate_sp_model
import sentencepiece as spm

def getData():
    # filename = '.data/multi30k/train.en'
    # generate_sp_model(filename, vocab_size=8000, model_type='bpe', model_prefix='zaid_sp_model')
    # #s = spm.SentencePieceProcessor(model_file='zaid_sp_model.model')
    # print(vars(s))
    # print(dir(s))
    # print(s.vocab_size())
    # print(s.bos_id())#exit()
    # print(s.eos_id())
    # print(s.unk_id())
    # print(s.pad_id())

    #exit()
    sp_gec = load_sp_model("BPE/GCEBPE30k.model")
    #sp_gec = load_sp_model("zaid_sp_model.model")
    # sp_gec =s
    # print(dir(sp_gec))
    # print(vars(sp_gec))
    #exit()
    src_pad_idx = sp_gec.pad_id()       #english_vocab.stoi["<pad>"]
    print("pad_index = ", src_pad_idx)
#    print("pad = ", sp_gec.decode(src_pad_idx))
    #exit()
    
    # print("print(len(sp_gec)) 1", len(sp_gec))
    # print(vars(sp_gec))
    # print(dir(sp_gec))
    #exit()

    bpe_field = Field(use_vocab = False, tokenize = sp_gec.encode, 
	    init_token = sp_gec.bos_id(), eos_token = sp_gec.eos_id(), pad_token = sp_gec.pad_id(), unk_token = sp_gec.unk_id(), batch_first = True)


    print("-----------------------------------------------")
    #print(TabularDataset.splits.__doc__)
    #tv_datafields = [("ignore", bpe_field), ("trg", bpe_field), ("src", bpe_field)]
    # train_data, valid_data, test_data = TabularDataset.splits(path = "/data/chaudhryz/ankit", train = "test10k.tsv", 
    #                                         validation="test10k.tsv", test = "test10k.tsv", format='tsv', skip_header=False, fields = tv_datafields)

    tv_datafields = [("trg", bpe_field), ("src", bpe_field)]
    train_data, valid_data, test_data = TabularDataset.splits(path = ".data/multi30k", train = "train.tsv", 
                                            validation="val.tsv", test = "test2016.tsv", format='tsv', skip_header=False, fields = tv_datafields)

    # train_data, valid_data, test_data = Multi30k.splits(
    #     exts=(".ennsw", ".en"), fields=tv_datafields,
    #     train='train',
    #     validation='val',
    #     test='test2016',
    #     path = '.data/multi30k'
    # )
    print(train_data)

    return sp_gec, train_data, valid_data, test_data
