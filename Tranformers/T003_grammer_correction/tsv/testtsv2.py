import spacy
from torchtext.data import Field, BucketIterator

from torchtext import data

#original_sentence	edited_sentence	edited	numchanges
# create Field objects
ORIGINAL_SENTENCE = data.Field()
EDITED_SENTENCE = data.Field()
EDITED = data.Field()
NUMCHANGES = data.Field()

# create tuples representing the columns
fields = [
  ('original_sentence', ORIGINAL_SENTENCE),
  ('edited_sentence', EDITED_SENTENCE),
  ('edited', EDITED), # ignore age column
  ('numchanges', NUMCHANGES)
]

# load the dataset in json format
test_ds = data.TabularDataset.splits(
#   path = '/data/chaudhryz/uwstudent1/2017-data/',
   path ='.',
   test = 'test2.tsv',
   format = 'tsv',
   fields = fields,
   skip_header = True
)

print(test_ds[0].examples[1].original_sentence)
print(test_ds[0].examples[1].edited_sentence)
#print(dir(test_ds[0].examples[1]))
