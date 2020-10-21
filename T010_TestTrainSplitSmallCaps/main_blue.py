from nltk.translate.bleu_score import sentence_bleu


def computeBLEU(ref, cand):
	ref = ref.split(" ");
	cand = cand.split(" ")
	# print(ref)
	score = sentence_bleu([ref], cand)
	return score


reference = 'I loved doing this'
candidate = 'I doing loved this';
score = computeBLEU(reference, candidate)
print(score)