
main.py              -- to run the LSTM based pointer network
main_datagen_test.py -- to generate dataset, and interpret it. (by converting ints back to words)

- All words in words.txt are of length 6 (6 is the number of features - ascii for each alphabet)
- If we want to sort variable length words alphabetically, we can use a special padding character
  say &, and assign ascii code one more than z. (so that python sorting function of sortWords()
  , i.e., (key=lambda e: e[0]), can work properly also.
- If we use some other embedding, we will automatically get a fixed length embedding for each word,
  e.g spacy vectors, but probably, those features won't be good for alphabetic sorting.

1. origList = generateWords(n) --- n is the sequence length (different words)
    # generates n words, return array of array of the form:
    # origList = [['wallow', 0], ['elnora', 1], ['amount', 2], ['demurs', 3]]

2. sortWords(origList)
    # we sort them w.r.t the first value in the array
    # sortList = [['amount', 2], ['demurs', 3], ['elnora', 1], ['wallow', 0]]

3. x, y = prepareInputForPtrNet(origList, sortedList)
    # input = [x[0] for x in origList]
    # target = [x[1] for x in sortedList]

4. x, y = batch(N)
   e.g, returns a batch of N=5 sequences, each seq of length say 8, and each word has 6 features,
   - dim of x would be torch.Size([5, 8, 6]), and
   - dim of y would be [5, 8]

   - x is of shape (BATCH_SIZE, SEQ_LENGTH, EMBEDDING/FEATURE_SIZE)

