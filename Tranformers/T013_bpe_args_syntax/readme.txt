
- checkin the code in Screbendi
- Run on Scribendi dataset
- Integrate context vector
    - run ankit's model on one of the public datasets
- save the sentence piece object and load it again as pkl so that we can test on local
    - runs but after x epochs throws => RuntimeError: CUDA error: device-side assert triggered
        - write code to filter out long sequences in BPE
- Figure out the issue why the saved model gives better results.
    - Cleanup code, and understand it better
- make diagram of pointer networks LSTM
- Make a pointer network from Transformer (Encoder only, and Encoder plus looped back decoder)

- how to train/make BPE model
- Get BPEmodel working on local system on multi30k (preferably on latest pytorch/torchtext)
- understand Batching, 

Command-line options:

---------------------------------------------------
meeting points
- Integrated BPE into my code
- Ran the BPE model on smaller multi30k dataset and it worked, though apparently, longer trainings are required.
  > got comparable results as earlier



# test2.py runs the testing
CUDA_VISIBLE_DEVICES=3 python test2.py --path multi30k --trainFile train.tsv --valFile val.tsv --testFile test2016.tsv
CUDA_VISIBLE_DEVICES=3 python test2.py --path scribendi --trainFile train4m.tsv --valFile test10k.tsv --testFile test10k.tsv 
# main.py runs the training
CUDA_VISIBLE_DEVICES=3 python main.py --path scribendi --trainFile train4m.tsv --valFile test10k.tsv --testFile test10k.tsv
CUDA_VISIBLE_DEVICES=3 python main.py --path multi30k --trainFile train.tsv --valFile val.tsv --testFile test2016.tsv

28.73,  22.66,   27.53,  20.11
35.88,  32.89   no eval

41.43,  33.02
41.43,  33.02
41.43,  33.02

36.77,  30.36    no eval

39.29,  30.02,   34.94,  27.55 nwe train
43.03,  34.66

44.35,  31.37,   37.16,  28.76
44.28,  30.69

46.43,  31.78,   43.75,  28.86
46.70,  33.10


36.22,  24.66,   32.77,  23.79
28.08,  20.96,   26.81,  20.09
49.15,  35.76

47.10,  31.38,   42.22,  30.88
46.26,  36.78

48.40,  32.27,   44.04,  29.95

40.93,  27.97,   37.56,  27.10
49.85,  35.49
Train Bleu score_train 100.00,  27.00
Train Bleu score_train 97.51,  41.68
Train Bleu score_train 98.65,  38.35

97.83,  42.34
91.83,  42.00
91.83,  42.00

Train Bleu score1 93.62
Test Bleu score2 41.11

** I currently have to "git push origin HEAD:master" and "git pull origin master" **