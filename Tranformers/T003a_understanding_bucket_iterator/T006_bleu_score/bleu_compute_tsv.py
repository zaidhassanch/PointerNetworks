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

# f = open("test10k.tsv", "r")
f = open("/data/chaudhryz/ankit/train300k.tsv", "r")

BleuScores = []

targets = []
outputs = []

for i, line in enumerate(f):
    orig = line.split("\t")[0]
    targ = line.split("\t")[1]
    # print(orig)

    tokens_orig = [token.text.lower() for token in spacy_eng(orig)]
    tokens_targ = [token.text.lower() for token in spacy_eng(targ)]

    outputs.append(tokens_orig)
    targets.append([tokens_targ])
    if i % 100 == 0:
        print(i)
print(bleu_score(outputs, targets))
