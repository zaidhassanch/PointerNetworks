import sentencepiece as spm
# s = spm.SentencePieceProcessor(model_file='test/test_model.model')

import sentencepiece as spm
path = ".data/multi30k/"

# train bpe encoder
# from Model Training in master/python/README.md and main readme usage instructions "Train SentencePiece Model"
spm.SentencePieceTrainer.train(input=path +'train.tsv',
                               model_prefix='zaid', vocab_size=29359, model_type = "BPE")

# why is there a limit to the vocabulary size
s = spm.SentencePieceProcessor(model_file='zaid.model')
print(s)
for n in range(20):
    # encode out_type allows to choose between string and int
    print(s.encode('New York boy ', out_type=str, enable_sampling=True, alpha=0.1))

print(s)