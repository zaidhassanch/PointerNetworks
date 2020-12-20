

This example does the language modeling.

Summary of Code:
1. Makes vocabulary from TRAINING TEXT FILE
   > tokenizes input file via the tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
   > vocab = build_vocab_from_iterator(map(tokenizer,....
   > vocab has itos, stoi, freqs, ...

2. PROCESS DATA to creates one dimensional indices of words (data_process)
   > train_data
   > test_data
   > val_data

3. Convert above data to prepare to get batch for while document
   > see explanation ** in main.py
   > 20 batch size for train data
   > 10 batch size for test and val data

4.