from configs import config

if config.USE_BPE == True:
    from data_modules.data_bpe import MyDataModule
else:
    from data_modules.data_spacy import MyDataModule

