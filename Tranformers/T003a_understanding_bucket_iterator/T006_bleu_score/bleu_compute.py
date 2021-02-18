from torchtext.data.metrics import bleu_score
import spacy

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

f = open("Feb17", "r")

targets = []
outputs = []

for i, line in enumerate(f):

    if (i-3) % 5 == 0:

        orig = line.strip().split(".", 1)[0]
        print(line.strip())
        print(orig)
        tokens_orig = [token.text.lower() for token in spacy_eng(orig)]

        outputs.append(tokens_orig)
    if (i-1) % 5 == 0:

        targ = line.strip().split(".", 1)[0]
        print(line.strip())
        print("+++++", targ)

        tokens_targ = [token.text.lower() for token in spacy_eng(targ)]

        targets.append([tokens_targ])





bleu = bleu_score(outputs, targets)

print(bleu)
