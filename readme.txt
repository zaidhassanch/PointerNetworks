
Understood the transformer architecture
-



T003_grammer_correction : it is basically the translation model.
        We can configure it for translation or word reordering

T005_reorder_words: we shuffled the words using tsv/fileshuffle.py
        and gave it the correct order english words. Did a good job

T006_reorder_pointer:
        - make *.ens as well as *.idx  (desired indexes)
        - 12:03-12:18 - search for code
        - 12:18-12:33 - tokenize sentence and check if size is the same
        - Ahmad I am
            2   0  1
            0   1  2
        -   I am Ahmad
            0  1   2
            1  2   0


- LSTM based pointer network block diagram
- What is our plan to replace


- Exploring Transformer implementation to find the right match to our problem
    - language modeling
    - machine translation
- Understood transformer architecture to quite an extent,
    Plan:
    - So probably take two days to complete my understanding
    - probably another 2-3 days to use transformer to order a sequence
    - I hope, training would be easier / faster on Transformers due to Parallism.
    - Then will work on Scribendi dataset.

    > how parallelism is used instead of sequential processing of LSTMs
    > why / how causal masking is used in its decoder
    > got some translation examples working,

===================================================
- Understood the Transformer architecture
  > How training can be done in parallel via causal masking
  > Self attention and "cross" attention
  > Positional encoding

- Understood a translation example which translated German to French
  > Automatic embedding generation in the network
  > Automatic positional encoding without using sin and cos functino
  > Multi-layered encoders and decoders

- Tried to run the above Translation Transformer architecture as a pointer network
  > Basically made a vocabulary of indices, each token corresponding to the shuffled position
  > The transformer network apparently got confused and could not train

- My father also got interested in our problem, and we do different things together

- After failed to train the transformer network as a pointer network
  > I thought that if it can do a good job in translation, why can't it handle reordering
  > So, I shuffled the words at input and gave the correct sentence as output. Bingo, it worked !

- If it can do translation and word reordering, why can't it introduce stop words. They don't carry meaning anyway ;-)
  > Bingo, it did a fairly good job.

- Reluctantly, I removed stop words, and shuffled the "meaningful words"
  > To my surprise, that worked too.
  > Achieved 95 blue score on training set, and roughly 35% on test set.
  > With proper hyperparameter tuning, and probably better datasets, things might get even better.

- Pruned the the architecture significantly:
  > Removed positional encoding completely
  > removed dropout completely
  > Removed multi-layered encoder and decoder, and used single encoder and decoder layer
  > number of heads reduced from 8 to 2
  > Even removed normalization and linear layers of encoder and decoder, still very good results.

- It is so amazing that such a simple PRUNED architecture, still does a great job.
  Example sentences:
  src   :>>> ['stove', 'food', 'man', 'cooking', '.']
    target:    ['a', 'man', 'cooking', 'food', 'on', 'the', 'stove', '.']
    pred  :    ['a', 'man', 'cooking', 'food', 'on', 'a', 'stove', '.']

- Did study scribendi dataset a little, plan to do use it.
      > it is a huge dataset, and the system really has hard time loading the dataset
      > Need to modify the input / output system of my code to read such a large dataset
      > May be read the dataset in sort of different batches.
      > Hints to manage it can be helpful (my code is written in pytorch, and I am using built in function)

- Since understanding of transformers is little better now, I can try again to build a pointer network
  > though, with current results of stop words insertion and reordering, this doesn't seem necessary
  > Probably, using just the encoder should be sufficient I think to make a pointer network
    -- by introducing couple of linear layers and a softmax which outputs indices
    -- or by a similar LSTM architecture which initializes LSTM decoder states to zero and then feeds back that state.

===============================================================
#ssh -N -f -L localhost:7774:localhost:7777 ahmad@chiisb1.nayatel.net -p8989


#ssh -L localhost:7773:localhost:7777 ahmad@chiisb1.nayatel.net -p8989

on remote: jupyter notebook --no-browser --port=7777

get the token from server, eg: ?token=538d89603dd0cfc477d14c97d13ead3b386d17523d2dee70

on local: ssh -N -f -L localhost:7773:localhost:7777 ahmad@chiisb1.nayatel.net -p8989


open : localhost:7773/?token=538d89603dd0cfc477d14c97d13ead3b386d17523d2dee70
sftp://chiisb1.nayatel.net:8989/



=============================================
- Test train split HP
	> Have my own sentences which it has never seen
- Compute accuracy HP
- I am sure he will talk to me

- Fully connected explore - Low priority
- Try on Urdu

T001_sort               : Original code
T002_sort_alphabet      : Code for word sorting
T003_sortTwoNumbers     : Extention of T001 to vector of features
T004_sortTwoNumbers     : Cleaned up code of T003
T005_sort_alphabets     : Try to merge T002 and T005


PHASE 1: Sort scalars
================================
- I Sorted 10 numbers
- Then sorted 100 numbers

PHASE 2: Sort Vectors
=====================
- Added provision for vectors and sorted vectors w.r.t. sum of vectors

PHASE 3: Sort Words Alphabetically,
===============================================================
- use ascii code as features
- Now since I could sort sum of vectors w.r.t a criteria, I should be able to sort fixed length words
- Extend to words of any length < N,
  > append a special character at the end of the word which can be treated it as <a, or >z

PHASE 4: Sort Jumbled sentences
======================================
- Instead of simple ascii representation of words, use "embedding" representation of words
- This would enable better sorting
