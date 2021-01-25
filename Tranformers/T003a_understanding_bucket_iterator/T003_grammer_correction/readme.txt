train_data  29001
valid_data  1015
test_data  1000
src vocabulary size:  7855
trg vocabulary size:  5895

spacy provides tokenizers,
Tokenizers process input files to generate vocabulary


ALPHA) build_vocab
    - returns vocab
    A) Counter update takes the words and updates the frequencies
        - counter.update(tokenizer(string_.lower()))
    B) Vocab:
        Attributes:
            freqs: A collections.Counter object holding the frequencies of tokens
                in the data used to build the Vocab.
            stoi: A collections.defaultdict instance mapping token strings to
                numerical identifiers.
            itos: A list of token strings indexed by their numerical identifiers.
        1) simply stored the collections.Counter object
            - self.freqs = counter
            - # stoi is simply a reverse dict for itos

        2) self.itos.append(word)
            - In Vocab """# sort by frequency, then alphabetically""" should be """# sort alphabetically, then by frequency"""
            - after sorting by frequency the words are simply appended

        3) stoi is simply a reverse dict for itos
            - self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

BETA) data_process
    - read each line in the input dataset
    - convert it into numbers
        - torch.tensor([eng_voc[token] for token in eng_tok(raw_en)],
                                                  dtype=torch.long)


GAMMA) DataLoader magically takes a list of tuples to make things right
        - DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=generate_batch)






Two young, White males are outside near many bushes.



>>  ['zwei', 'junge', 'weiße', 'männer', 'sind', 'im', 'freien', 'in', 'der', 'nähe', 'vieler', 'büsche', '.']
    ['two', 'young', ',', 'white', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.']
>>  ['mehrere', 'männer', 'mit', 'schutzhelmen', 'bedienen', 'ein', 'antriebsradsystem', '.']
    ['several', 'men', 'in', 'hard', 'hats', 'are', 'operating', 'a', 'giant', 'pulley', 'system', '.']
>>  ['ein', 'kleines', 'mädchen', 'klettert', 'in', 'ein', 'spielhaus', 'aus', 'holz', '.']
    ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']