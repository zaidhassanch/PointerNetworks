import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

"""
To install spacy languages do:
python -m spacy download en
python -m spacy download de
"""

spacy_ger = spacy.load("de") 
spacy_eng = spacy.load("en")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]



def getData():
    german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

    english = Field(
        tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
    )


    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".de", ".en"), fields=(german, english)
    )
    
    german.build_vocab(train_data, max_size=10000, min_freq=2)
    english.build_vocab(train_data, max_size=10000, min_freq=2)

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


    return german, english, train_data, valid_data, test_data