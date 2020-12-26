
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
    - Then will work on Srabundy dataset.

    > how parallelism is used instead of sequential processing of LSTMs
    > why / how causal masking is used in its decoder
    > got some translation examples working,






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
