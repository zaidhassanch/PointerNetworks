
- recover the version whose results we quoted in our scribendi meeting (6 layers)
  with blue scores. ESTABLISH OUR BASELINE
- parallelize the translate sentence (inference)
- May be look at Ankit code.
  > What are the stepsizes, layers sizes, convergence speed
  > How is he parallelizing the inference

AHMAD
=====
- Establish base data set(s), Multi30k, JaneErye, Public,
- Compare convergence BPE vs Spacy for Trained Embed
  > Make work around to modify dataset so that spacy tokenization works for Trained embed.
- Introduce Spacy vectors (no training for embedding)

https://vectorinstitute.zoom.us/j/7499157050?pwd=dWV6UGRoUGhaWVRpTkNRNHZnOEgzQT09

Step 1:
    - Original attention in all three places- encoder, decoder and cross
    - confirm: 16, 59
Step 2:
    - kq code on all three (where there are no Wk and Wq)
    - save kq in all
    - try on 2 layers
    - qkT softmax will be identity and for cross attention it will be correct

=========================================================================
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
            - In Vocab """# sort by frequency, then alphabetically""" should
               be """# sort alphabetically, then by frequency"""
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

==================================================


Two young, White males are outside near many bushes.



>>  ['zwei', 'junge', 'weiße', 'männer', 'sind', 'im', 'freien', 'in', 'der', 'nähe', 'vieler', 'büsche', '.']
    ['two', 'young', ',', 'white', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.']
>>  ['mehrere', 'männer', 'mit', 'schutzhelmen', 'bedienen', 'ein', 'antriebsradsystem', '.']
    ['several', 'men', 'in', 'hard', 'hats', 'are', 'operating', 'a', 'giant', 'pulley', 'system', '.']
>>  ['ein', 'kleines', 'mädchen', 'klettert', 'in', 'ein', 'spielhaus', 'aus', 'holz', '.']
    ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']