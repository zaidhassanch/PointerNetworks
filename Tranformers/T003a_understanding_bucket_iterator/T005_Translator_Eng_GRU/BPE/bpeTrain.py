import sentencepiece as spm

spm.SentencePieceTrainer.train(input='.data/multi30k/train.en', model_prefix='english'
                                      , vocab_size=10000, model_type='bpe')
# spm.SentencePieceTrainer.train(input='.data/multi30k/train.de', model_prefix='german',
#                                vocab_size=10000, model_type='bpe')