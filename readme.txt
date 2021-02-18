- I needed some results which I can reproduce and then improve.
  > Probably, the new pipelines Ankit has created can be used now.
  > But, still they are very slow
  > So, I feel a little tied up.

- We are working on transforming sentence from "sentence space" to "context" and "syntax" space and then back
- I thought why not convert to another sentence space (ie another language and then back)
- Tried it using google translate (translated to german and back)
- And have some interesting results.


15 mins targets
==============================
12:27-12:42 - open bleu score computation on Scribendi server
            - Achieved: Not sure
12:42-12:57 - local averaging of bleu score (make a list and then average it)
                - Note: list will be long
            - Achieved
                - stuck on disappearing code in bleu compute
                    - Note: Feb 17 has been transmitted as is 
12:57-1:12 - take averaging to scribendi server and start computing results


15 mins targets
=================================================================
11:52-12:02 - Planning/ Find the code I want to work on
                - decided to use the transformer code for fast training
                - find the transformer code and
                   - probably T004 with transformer selected
12:02-12:17 - Take tsvs of fce and train on them

bleu score bad, Scribendi,Zaid


Today's task is to fix "is/are" problem
- filter out sentences which have is/are problem
- We should have a method to test how much % have we fixed, not only 4 sentences
  > sentence comparator
    - function compareSents(sent1, sent2)
        - length
        - compare word by word and print difference

  > read in sentences from file
  > How many should I test after each epoch, and how to evaluate on full test/train dataset
  > First do serially, one by one
  > how to evaluate it in parallel (low priority)
- Should have test / train dataset, so that we can test on unseen data
- Then go inside code to figure out why issue is still present

- we might have to see how many sentences are we accurately reproducing.
- We might think of using spacy embeddings (should definitely produce better results)
============================================


- Add BPE
   why bpe is good, how did somebody invent bpe
- change lstm encoder with tranform encoder
- make balanced dataset
- think about error types and add sentences with those errors
- Learn NLTK
- How can NLTK help us generate more data (wrong sentences)


- Try to make a mechanism for generating datasets
  - Find lot of sentences which are correct
  - Change was to were, is to are, ...
  - fix them via transformer

- Categorize is as PRESENT=0, was as PAST=1
  - Use category as input and make a sentence out of list of words (missing is or are, or were, etc)

- More on data generation
  During this:
     - Learn NLTK
     - We want to generate known errors, e.g.

          Sentence generator with input syntax (1,2,3,4)
          Input: Horse sit under tree
          Output: 1 The Horse is sitting under a tree
                  2 The Horse will sit under a tree
                  3 The horse was sitting under a tree
                  4 The horse has been sitting under that tree

          The preceding module generates the context
                  The horse are sits under a tree


- Try out public datasets for grammar correction.

send results for different models for Multi-30k dataset

- Integrate BPE encoding
- Try out or scribendi
- Zaid should explain what is context/syntax and Ahmad/zaid will try to figure out steps to implement.

Speeding up the code:
- See issue in dataloader, why slower than bucket iterator
- Learn pytorch-lightening, allennlp
-

- INCORPORATE ANKIT'S OTHER MODELS IN OUR SYSTEM
- SEE HOW HE IS TESTING IN PARALLEL TO CALCULATE BLUE SCORE
- 

Meeting with Ankit/Pascal (Feb 10)
1) I looked at Ankit's code and he already has different variants of architectures implemented
	- these architectures include
		- simple GRU, which is baselie
		- GRU with attention
		- Baseline plus Transformer integration (Check if syntax code is being used)
		- Ankit asked me to replace his tranformer by my transformer and I have done that.

      I have a very important, my obseration may be wrong:
              - Translate sentence is OK
              - But  in EVALUATE FUNCTION The target sequence is also input to the model/tranformer
              - We need to input only <sos> token and process the sentence in a loop

		- Before I consume all GPUs, I would like to see the baseline results.
		    Otherwise, we will be wasting GPU power unnecessarily.
        - To debug/integrate into Ankit's code, I am using Multi30k so that I am not limited by Scribendi GPUs

    - Integrated my transformer into Ankit's code and checked in. (no training done yet, just sanity check no bugs)
    - Drew the architecture diagram for easy references
      > instead of the way current diagrams were drawn, I have redrawn as proper system diagrams with inputs / outputs
        I think this is a slightly better representation of our architecture

----------------------------------------------------------
Meeting with Dr Pascal (Feb 4, 2021)
- I was supposed to start integration of my probe into ankit's code
  > ankit promised to sent some tips on how to integrate.
  > he then later emailed me that it is going to take a while.

- Anyways, Didn't really wait for it
  > Getting hands on it now, probably will be able to start integration soon
  > To speed up development and testing, I am using Multi30k dataset
     -- much easier to work on my machine
  > I am developing a clean code so that integration is easier
  > in the process I have an observation:
    BPE tokenizer is very slow and increases probably 50% of training time.
    I have Yet to confirm this observation,
    Tokenizing dataset in advance can probably speed up training
  > I need to discuss context / syntax partitioning in the Exp10 code with Ankit
    Right now the things do not look correct. Since it is just a simple splitting of
    hidden states. Hidden states are concat and passed through fc and splitted
    does not make sense to me
  > Experimented a little with adding attention, results are not final yet, but I see some improvement.

  Now I have models for Seq2Seq training.
     > Transformer                     1    very fast, and after 1 epoch gives some good (30 sec)
     > GRU based model without atten   3    fast, and after 7 epoch gives some good (40 sec)
     > GRU based model with atten      2    fast, and after 2 epoch gives some good (65 sec)

- Few more point: Some deprecated stuff is being used
   - BucketIterator, Field, both are deprecated.
   - Explored new methods of data loading
   - BucketIterator vs data loader
   - Dataloader is working but somewhat slower but still exploring, I am sure, I am missing something

- Prof Pascal - profile code
  - made an example of simple counter and observed results
  - code which would otherwise take .25sec, can take 4 sec with prof - intruisive profiling
    #but we know where the time is being spent. We should scale interpretation
  - Right now profiled function, yet to profile class functions, haven't tried yet.
  - Still more work required on this

==============================================================

- Now, I have started understanding the code myself
  - getting significant hand on it, probably a little more time to start integration
  - Still need to see good results on the model
  - Too hard to work on Scribendi dataset since it needs lot of training time
     > training time and restriction on not to copy is a little hurdle
     > To remove this hurdle, I switched back to Multi30k,
       Made it to work on Multi-30k. (BPE is obviously trained for english)
  - In the process of making a framework which makes integration easier
       > it currently uses SPacy tokenizer, have to shift it to BPE also
       > Right now not in a very presentable form
       > SOMEHOW, MY FRAMEWORK DOES FASTER ITERATION (MORE THAN 50% FASTER)
         - in my framework I am using spacy for tokenization, which appears to be the reason
           as opposed to BPE
         - The data I am using is my reference german to english translation dataset since I
           can't use scribendi dataset


- I need to discuss context / syntax partitioning in the Exp10 code with Ankit





https://github.com/bentrevett/pytorch-seq2seq
https://towardsdatascience.com/a-comprehensive-guide-to-neural-machine-translation-using-seq2sequence-modelling-using-pytorch-41c9b84ba350

- German to English via Ankit code
- Add attention in Ankit code
- See timings of Ankit code
- Use BPE in our LSTM code
- Draw architectures of Ankit's code


- convert to seq2seq
- evaluate with horse german sentence
- use german


- run code
- understand what ankit is saying
- profile code
    > This is intruisive profiling
      - made an example of simple counter and observed results
      - code which would otherwise take .25sec, can take 4 sec with prof
        but we know where the time is being spent. We should scale interpretation

- adversarial loss in text generation


- read papers from ankit



- Basics of some tools of pytorch for nlp

- Understand BucketIterator and Dataloader difference.
    -

  - some more understanding of Dataloader required, collatefn, etc.
- Tried to train, even reduced the dataset, but trainings give zero blue score on test set.
- Currently, trying to make a pointer network using transformer.


==================================
- can use simple attention and some linear layer -- implemented successfully, but didn't work
- copy TransformerEncoderlayer and simplify it   -- didnt't feel like
===================
- apparently, our tranformer T006_understand works -- figure out how we made it work. Extend to word ordering.
===================

- Try to understand Adversarial screenshot
- Understand and Train BPE
- Try pointer generator network
- Draw the network diagram for LSTM based pointer network
  > try to write down the dimensions
  > study (ho, co), ...
- Study the attention layer, why it is working ------ IMPORT
- How to integrate Transformer decoder

- why simple linear layers don't work.
Explanation of how LSTM based pointer network works is given in : T006_integrate_words_ptrnet/readme.txt
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
