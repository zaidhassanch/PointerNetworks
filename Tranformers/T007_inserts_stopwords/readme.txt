
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


target:    ['i', 'would', 'like', 'build', 'upon', 'this', 'experience', 'in', 'my', 'future', 'activities', '.']
pred  :    ['i', 'am', 'my', 'future', ',', 'and', 'experience', 'to', 'my', 'future', 'to', 'my', 'future', 'experience', '.']
src   :>>> ['goal', 'campaign', 'saudi', 'completely', 'reliant', 'solar', 'energy']
target:    ['our', 'goal', 'in', 'this', 'campaign', 'is', 'that', 'by', '2030', ',', 'saudis', 'will', 'be', 'completely', 'reliant', 'on', 'solar', 'energy', '.']
pred  :    ['<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>', ',', '<unk>']
src   :>>> ['content', 'marketing', 'helped', 'quickly', 'improve', 'relationship', 'existing', 'customer', 'improved', 'retention', 'nicely']
target:    ['content', 'marketing', 'helped', 'to', 'quickly', 'improve', 'our', 'relationships', 'with', 'existing', 'customers', 'and', 'this', 'improved', 'our', 'retention', 'nicely', '.']
pred  :    ['the', 'relationship', 'of', 'the', 'relationship', 'between', 'the', 'relationship', 'and', '<unk>', 'relationship', 'to', 'the', 'relationship', 'of', 'the', 'relationship', '.']
src   :>>> ['na']
target:    ['(', '9', ')', 'p.eleph', '.']
pred  :    ['<unk>']
src   :>>> ['cure', 'knee', 'treatment', 'directed', 'symptomatic', 'relief']
target:    ['there', 'is', 'no', 'cure', 'for', 'knee', 'oa', ',', 'and', 'treatment', 'is', 'directed', 'at', 'symptomatic', 'relief', '.']
pred  :    ['<unk>', 'treatment', 'treatment', 'treatment', 'treatment', '.']
src   :>>> ['na']
target:    ['159.47']
pred  :    ['<unk>']
src   :>>> ['classifies', 'veto', 'player', 'two', 'independent', 'group']
target:    ['he', 'classifies', 'veto', 'players', 'into', 'two', 'independent', 'groups', '.']
pred  :    ['the', '<unk>', 'of', 'two', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '.']
Test Bleu s