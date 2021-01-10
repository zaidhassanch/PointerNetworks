# import os

# import spacy
# import pickle
from torchtext.data import Field
from torchtext.datasets import Multi30k #, TranslationDataset
from torchtext.data.functional import load_sp_model, generate_sp_model

def getData():
    filename = '.data/multi30k/train.en'
    generate_sp_model(filename, vocab_size=8000, model_type='unigram', model_prefix='zaid_sp_model')


    #exit()
    #sp_gec = load_sp_model("BPE/GCEBPE30k.model")
    sp_gec = load_sp_model("zaid_sp_model.model")
    print(dir(sp_gec))
    # print(vars(sp_gec))
    # exit()

    
    print("print(len(sp_gec)) 1", len(sp_gec))
    print(vars(sp_gec))
    print(dir(sp_gec))
    # exit()

    bpe_field = Field(use_vocab = False, tokenize = sp_gec.encode, 
	    init_token = sp_gec.bos_id(), eos_token = sp_gec.eos_id(), pad_token = sp_gec.pad_id(), unk_token = sp_gec.unk_id(), batch_first = True)

    tv_datafields = [("src", bpe_field), ("trg", bpe_field)]

    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".ennsw", ".en"), fields=tv_datafields,
        train='train',
        validation='val',
        test='test2016',
        path = '.data/multi30k'
    )
    print(train_data)

    return sp_gec, train_data, valid_data, test_data
