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

================================================================
- figure out how to speedup seq2seq, we see that in his code it takes 92sec, vs 50 sec.

Zaid (Dataloader, BPE, TSV, GRU, TRansformer, LSTM with ATTn).
- use TSV (using our old code complete)
  > BUT SHOULD USE TABULARDATA
- loading of data from TSV appears to be very fast
- use BPE (ability to switch between spacy and BPE) - 30 min
- draw arch diagram for ankit code
- add the pytorch eng-fra translator DrPascal/NLP_tutorials/003_myMTSS
- use data from m2 files (m2 -> tsv, tsv -> m2)
- Use m2 files to see how well it corrects grammer of standard datasets -- V IMP

PROFILING:
==========
> Understood profiling (ine profiling for functions, not for classes yet).

- Re-discuss parallelism in Transformers.
  > seq2seq is faily fast now.

- Run ankit code on Scribendi and see results

--------------------------
- Architecture draw
- understood ankit code
- profiling
- implemented ankit code in our code (removed unk issue)
- fixed blue score in dataloader case