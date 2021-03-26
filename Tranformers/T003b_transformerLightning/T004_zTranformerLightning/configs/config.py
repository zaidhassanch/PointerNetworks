BPE_FROM_PICKLE = True
BPE_PATH = "GCEBPE4m.model"

PYTORCH_TRANSFORMER = False
SELF_ATTN = "ORIGINAL"  #"ORIGINAL"|"SUMMARIZED"|"OUR"|"CROSS"
LOAD_NEW_METHOD = True
USE_BPE = True
MAX_LEN = 500
GPUS = 1
MAX_EPOCHS = 15
PRECISION = 32 #32|16

N_LAYERS = 1
NUM_HEADS = 1
FORWARD_EXP = 1

COMPUTE_BLEU = True

SEED = 1234

TRAIN_SRC = 'tsv/sent_9_jane_eyre.src'
TRAIN_TGT = 'tsv/sent_9_jane_eyre.tgt'
VAL_SRC = 'tsv/sent_9_jane_eyre.src'
VAL_TGT = 'tsv/sent_9_jane_eyre.tgt'
TEST_SRC = 'tsv/sent_9_jane_eyre.src'
TEST_TGT = 'tsv/sent_9_jane_eyre.tgt'

sentences = []
sentences.append("What, you is a baby after all!")
sentences.append("You is afraid of ghosts?")
sentences.append("But is your relatives so very poor?")
sentences.append("Well, Jane Eyre, and is you a good child?")