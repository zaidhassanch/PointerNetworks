BPE_FROM_PICKLE = True
BPE_PATH = "GCEBPE4m.model"

PYTORCH_TRANSFORMER = False
SELF_ATTN = "ORIGINAL"  #"ORIGINAL"|"SUMMARIZED"|"OUR"|"CROSS"
LOAD_NEW_METHOD = True
USE_BPE = False
MAX_LEN = 500
GPUS = 4
MAX_EPOCHS = 15
PRECISION = 16 #32|16

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
sentences.append("Ein kleiner Junge im Fußballdress hält die Hände vors Gesicht und weint.")
sentences.append("Ein Mann macht Werbung mit einem riesigen Schild, das auf sein Fahrrad gebunden ist.")
sentences.append("Eine Frau steht auf einem grünen Feld, hält einen weißen Hund und zeigt auf einen braunen Hund.")
# sentences.append("ein mann, der rotes hemd tragt, das unter einem baum sitzt.")
# sentences.append("ein hund, der einer katze nachlauft, um sie zu schlagen.")
# sentences.append("ein alter mann, der versucht, von einem kaputten stuhl aufzustehen.")
# sentences.append("A horse is walking beside a boat under a bridge.")
# sentences.append("Two men are removing tree branches.")
# sentences.append("A young boy in a red life jacket is swimming in a pool.")
# sentences.append("Two kids are swinging on a playground.")