#from nltk.translate.bleu_score import corpus_bleu
from torchtext.data.metrics import bleu_score
references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
candidates = [['this', 'is', 'a', 'test']]
score = bleu_score(candidates, references)
print(score)