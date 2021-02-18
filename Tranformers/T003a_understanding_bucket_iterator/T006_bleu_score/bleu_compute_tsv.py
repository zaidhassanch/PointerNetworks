from torchtext.data.metrics import bleu_score
import spacy
import numpy as np

spacy_eng = spacy.load("en")
#
# orig = "When you go downhill, you have to stick out your chest or you will fell down."
# targ = "When you go downhill, you have to stick out your chest or you will fall down."
#

#
# orig = spacy_eng(orig)
#
# print(tokens_orig)

# bleu = bleu_score([[tokens_orig]],[tokens_orig])
#
# print(bleu)

f = open("test10k.tsv", "r")

BleuScores = []

targets = []
outputs = []

for i, line in enumerate(f):
    orig = line.split("\t")[0]
    print(orig)
#
#     orig = line.strip().split(".", 1)[0]
#     print(line.strip())
#     print(orig)
#     tokens_orig = [token.text.lower() for token in spacy_eng(orig)]
#
#     outputs.append(tokens_orig)
#
#     bleu = bleu_score([outputs[-1]], [targets[-1]])
#     bleu_overall = bleu_score(outputs, targets)
#     BleuScores.append(bleu)
#     # if np.int(np.floor((i + 1) /5)+ 1) % 2 == 0:
#     #     # bleu = bleu_score(outputs, targets)
#     #     print(np.int(np.floor((i + 1) /5)+1))
#     #     bleu = bleu_score(outputs, targets)
#     #     BleuScores.append(bleu)
#     #     outputs = []
#     #     targets = []
#
#     if (i - 1) % 5 == 0:
#         targ = line.strip().split(".", 1)[0]
#         print(line.strip())
#         print("+++++", targ)
#
#         tokens_targ = [token.text.lower() for token in spacy_eng(targ)]
#
#         targets.append([tokens_targ])
#
# BleuScores = np.array(BleuScores)
# print("average = ", np.mean(BleuScores))
# print(bleu_score(outputs, targets))
# print(BleuScores)
#
