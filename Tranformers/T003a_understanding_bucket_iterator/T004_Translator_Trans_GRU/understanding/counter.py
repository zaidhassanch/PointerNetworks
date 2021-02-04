from collections import Counter
from torchtext.vocab import Vocab
import io

c = Counter()                           # a new, empty counter

c = Counter('gallahad')                 # a new counter from an iterable

c = Counter({'red': 4, 'blue': 2})      # a new counter from a mapping

c = Counter(cats=4, dogs=8)             # a new counter from keyword args

s = ['a', 'quick', 'brown', 'fox', 'jumps', 'over', 'a', 'lazy', 'lazy', 'dog'];
c = Counter(s)
c.update(s)
print(c)

# ger_path = '.data/multi30k/train.de'
# raw_de_iter = iter(io.open(ger_path, encoding="utf8"))
#
# for sent in raw_de_iter:
#     print("-"+sent+"-")

a = [1,2,3,4,5]
b = [6, 7, 8, 9]

for a1, b1 in zip(a,b):
    print(a1, b1)