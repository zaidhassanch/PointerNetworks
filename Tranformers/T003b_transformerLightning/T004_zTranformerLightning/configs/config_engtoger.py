BPE_FROM_PICKLE = True
BPE_PATH = "GCEBPE4m.model"

PYTORCH_TRANSFORMER = True
SELF_ATTN = "OUR"  #"ORIGINAL"|"SUMMARIZED"|"OUR"
LOAD_NEW_METHOD = True
USE_BPE = False
MAX_LEN = 500
GPUS = 1
MAX_EPOCHS = 100

N_LAYERS = 6
NUM_HEADS = 8
FORWARD_EXP = 4

TRAIN_SRC = '.data/multi30k/train.en'
TRAIN_TGT = '.data/multi30k/train.de'
VAL_SRC = '.data/multi30k/val.en'
VAL_TGT = '.data/multi30k/val.de'
TEST_SRC = '.data/multi30k/test2016.en'
TEST_TGT = '.data/multi30k/test2016.de'

COMPUTE_BLEU = True

sentences = []
# sentences.append("Ein Pferd geht unter einer Brücke neben einem Boot.")
# sentences.append("ein mann, der rotes hemd tragt, das unter einem baum sitzt.")
# sentences.append("ein hund, der einer katze nachlauft, um sie zu schlagen.")
# sentences.append("ein alter mann, der versucht, von einem kaputten stuhl aufzustehen.")
sentences.append("A horse is walking beside a boat under a bridge.")
sentences.append("Two men are removing tree branches.")
sentences.append("A young boy in a red life jacket is swimming in a pool.")
sentences.append("Two kids are swinging on a playground.")