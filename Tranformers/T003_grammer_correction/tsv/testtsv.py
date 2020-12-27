import spacy
from torchtext.data import Field, BucketIterator

from torchtext import data
    
# create Field objects
AUTHOR = data.Field()
AGE = data.Field()
TWEET = data.Field()
LOCATION = data.Field()

# create tuples representing the columns
fields = [
  ('author', AUTHOR),
  ('location', LOCATION),
  (None, None), # ignore age column
  ('tweet', TWEET)
]

# load the dataset in json format
test_ds = data.TabularDataset.splits(
   path = '.',
   test = 'test.tsv',
   format = 'tsv',
   fields = fields,
   skip_header = True
)

print(test_ds[0].examples[0].author[0])