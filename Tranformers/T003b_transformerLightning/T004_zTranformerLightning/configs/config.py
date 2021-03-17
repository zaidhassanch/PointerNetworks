BPE_FROM_PICKLE = True
BPE_PATH = "GCEBPE4m.model"

PYTORCH_TRANSFORMER = False
SELF_ATTN = "SUMMARIZED"  #"ORIGINAL"|"SUMMARIZED"|"OUR"
LOAD_NEW_METHOD = True
USE_BPE = False
MAX_LEN = 500
GPUS = 1
MAX_EPOCHS = 10

N_LAYERS = 6
NUM_HEADS = 8
FORWARD_EXP = 4

TRAIN_SRC = '.data/multi30k/train.de'
TRAIN_TGT = '.data/multi30k/train.en'
VAL_SRC = '.data/multi30k/val.de'
VAL_TGT = '.data/multi30k/val.en'
TEST_SRC = '.data/multi30k/test2016.de'
TEST_TGT = '.data/multi30k/test2016.en'

COMPUTE_BLEU = False

sentences = []
sentences.append("Ein Pferd geht unter einer Brücke neben einem Boot.")
sentences.append("ein mann, der rotes hemd tragt, das unter einem baum sitzt.")
sentences.append("ein hund, der einer katze nachlauft, um sie zu schlagen.")
sentences.append("ein alter mann, der versucht, von einem kaputten stuhl aufzustehen.")
# sentences.append("A horse is walking beside a boat under a bridge.")
# sentences.append("Two men are removing tree branches.")
# sentences.append("A young boy in a red life jacket is swimming in a pool.")
# sentences.append("Two kids are swinging on a playground.")