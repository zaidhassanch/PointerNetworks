BPE_FROM_PICKLE = True
BPE_PATH = "GCEBPE4m.model"

PYTORCH_TRANSFORMER = False
SELF_ATTN = "ORIGINAL"  #"ORIGINAL"|"SUMMARIZED"|"OUR"|"CROSS"
LOAD_NEW_METHOD = True
USE_BPE = True
MAX_LEN = 500
GPUS = 4
MAX_EPOCHS = 15
PRECISION = 32 #32|16

N_LAYERS = 6
NUM_HEADS = 8
FORWARD_EXP = 4

COMPUTE_BLEU = False

TRAIN_SRC = 'dataFiles/train4m.src'
TRAIN_TGT = 'dataFiles/train4m.tgt'
VAL_SRC = 'dataFiles/test10k.src'
VAL_TGT = 'dataFiles/test10k.tgt'
TEST_SRC = 'dataFiles/test10k.src'
TEST_TGT = 'dataFiles/test10k.tgt'

sentences = []
#sentences.append("During counselling, therapist should aware the multi cultural context of each client and apply the skills according to the needs of the client.")
# Further , during counseling, the therapist should be aware of the multicultural context of the client and apply his or her skills according to the needs of the client.
sentences.append("The research was conducted at six CBE villages located on the three provinces in South Korea (see Figure 1.) .")
#The research was conducted in six CBE villages in three provinces in South Korea (see Figure 1) .
sentences.append("Initially , 690 participants were invited, 71 of these participants were declined or did not complete the questionnaires.")
# Initially , 690 participants were invited, but 71 of these participants were declined or did not complete the questionnaires.
sentences.append("The original rubric contain four areas for studying TPACK.")
# The original rubric contains four areas for studying TPACK.
sentences.append("The tie is the term used to explain the relationship between one node to another.")
# The tie is the term used to explain the relationship between one node and another.