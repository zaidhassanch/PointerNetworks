
To run English to English just change
    spacy_ger = spacy.load("de")
    spacy_eng = spacy.load("en")
and
    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".de", ".en"), fields=(german, english)
    )
 to
    spacy_ger = spacy.load("en") <<<<<<
    spacy_eng = spacy.load("en")

    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".en", ".en"), fields=(german, english)     <<<<<<
    )
==============
also change input sentences in main.py and train.py

sentence = ['ein', 'pferd', 'geht', 'unter', 'einer', 'brücke', 'neben', 'einem', 'boot', '.']
# sentence = ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']
