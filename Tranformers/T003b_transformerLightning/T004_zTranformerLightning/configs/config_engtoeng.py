BPE_FROM_PICKLE = True
BPE_PATH = "GCEBPE4m.model"

PYTORCH_TRANSFORMER = False
SELF_ATTN = "ORIGINAL"  #"ORIGINAL"|"SUMMARIZED"|"OUR"|"CROSS"
LOAD_NEW_METHOD = True
USE_BPE = False
MAX_LEN = 500
GPUS = 4
MAX_EPOCHS = 2
PRECISION = 32 #32|16
LEARNING_RATE = 3e-2

N_LAYERS = 6
NUM_HEADS = 8
FORWARD_EXP = 4

TRAIN_SRC = '.data/multi30k/train.en'
TRAIN_TGT = '.data/multi30k/train.en'
VAL_SRC = '.data/multi30k/val.en'
VAL_TGT = '.data/multi30k/val.en'
TEST_SRC = '.data/multi30k/test2016.en'
TEST_TGT = '.data/multi30k/test2016.en'

sentences = []
sentences.append("A horse is walking beside a boat under a bridge.")
sentences.append("Two men are removing tree branches.")
sentences.append("A young boy in a red life jacket is swimming in a pool.")
sentences.append("Two kids are swinging on a playground.")