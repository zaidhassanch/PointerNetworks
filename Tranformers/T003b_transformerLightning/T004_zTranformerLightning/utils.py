from configs import config

if config.USE_BPE == True:
    from utils_dir.utils_bpe import translate_sentence, computeBLEU, writeArrToCSV
else:
    from utils_dir.utils_spacy import translate_sentence, computeBLEU, writeArrToCSV