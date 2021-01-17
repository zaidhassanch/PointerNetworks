import torch
from torchtext.data import Field, BucketIterator

from data_loader import getData #, getData_newMethod

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", "-path", help="the path")
parser.add_argument("--trainFile", "-trn", help="the train file")
parser.add_argument("--valFile", "-val", help="the validation file")
parser.add_argument("--testFile", "-tst", help="the test file")
args = parser.parse_args()

batch_size = 1


#german_vocab, english_vocab, train_data, valid_data, test_data = getData_newMethod()
print("===============================before loading")
spe_dec, train_data, valid_data, test_data = getData(args.path, args.trainFile, 
                                                args.valFile, args.testFile)
print("train_data ", len(train_data.examples))
print("valid_data ", len(valid_data.examples))
print("test_data ", len(test_data.examples))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
    )

max_length = 0 
for batch_idx, batch in enumerate(train_iterator):
    # Get input and targets and get to cuda
    # print(batch_idx)
    inp_data = batch.src.permute(1,0)
    # print(inp_data.shape)
    
    inp_data = inp_data.to(device)
    target = batch.trg.permute(1,0)
    target = target.to(device)
    print("inp_data.shape", inp_data.shape)
    print("target.shape", target.shape)

    inputSeqLength = inp_data.shape[0]
    targetSeqLength = target.shape[0]

    print("inp_data, inp_data.shape", inp_data[:,0], inp_data[:,0].shape)
    translated_sentence_input = spe_dec.decode(inp_data[:,0])
    
    translated_sentence_target = spe_dec.decode(target[:,0])

    print("Input Sentence: ", translated_sentence_input)
    print("Target: ", translated_sentence_input)

    print("before")
    print (max_length, inputSeqLength, targetSeqLength)
    if inputSeqLength > max_length:
        max_length = inputSeqLength

    if targetSeqLength > max_length:
        max_length = targetSeqLength

    print("after")
    print (max_length, inputSeqLength, targetSeqLength)

    exit()

print("max_length: ", max_length)




