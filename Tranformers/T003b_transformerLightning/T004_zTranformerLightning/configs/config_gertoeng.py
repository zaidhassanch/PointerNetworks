BPE_FROM_PICKLE = True
BPE_PATH = "GCEBPE4m.model"

PYTORCH_TRANSFORMER = False
SELF_ATTN_OUR = True
LOAD_NEW_METHOD = True
USE_BPE = False
MAX_LEN = 500
GPUS = 4

TRAIN_SRC = '.data/multi30k/train.de'
TRAIN_TGT = '.data/multi30k/train.en'
VAL_SRC = '.data/multi30k/val.de'
VAL_TGT = '.data/multi30k/val.en'
TEST_SRC = '.data/multi30k/test2016.de'
TEST_TGT = '.data/multi30k/test2016.en'

sentences = []
sentences.append("Ein Pferd geht unter einer Brücke neben einem Boot.")
sentences.append("ein mann, der rotes hemd tragt, das unter einem baum sitzt.")
sentences.append("ein hund, der einer katze nachlauft, um sie zu schlagen.")
sentences.append("ein alter mann, der versucht, von einem kaputten stuhl aufzustehen.")