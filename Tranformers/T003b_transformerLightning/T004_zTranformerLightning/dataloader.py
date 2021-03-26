import torch
from collections import Counter
from torchtext.vocab import Vocab
import io
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from configs import config

def build_vocab(filepath, tokenizer):
  counter = Counter()
  with io.open(filepath, encoding="utf8") as f:
    for string_ in f:
      string_ = string_.rstrip()
      counter.update(tokenizer(string_.lower()))

  vocab = Vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'], max_size=40000, min_freq=2);
  vocab.init_token = '<sos>'
  vocab.eos_token = '<eos>'
  return vocab

def data_process(ger_path, eng_path, ger_voc, eng_voc, ger_tok, eng_tok):
  raw_de_iter = iter(io.open(ger_path, encoding="utf8"))
  raw_en_iter = iter(io.open(eng_path, encoding="utf8"))
  data = []
  for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
    raw_en = raw_en.lower().rstrip()
    raw_de = raw_de.lower().rstrip()
    de_tensor_ = torch.tensor([ger_voc[token] for token in ger_tok(raw_de)],
                            dtype=torch.long)

    en_tensor_ = torch.tensor([eng_voc[token] for token in eng_tok(raw_en)],
                            dtype=torch.long)

    # tokens = [token.lower() for token in eng_tok(raw_en)]
    # print(tokens)
    #
    # text_to_indices = [eng_voc[token] for token in tokens]
    # translated_sentence = [eng_voc.itos[idx] for idx in text_to_indices]
    # print(raw_en)
    # print(text_to_indices)
    # print(translated_sentence)
    # exit()
    data.append((de_tensor_, en_tensor_))
  return data

def getData2(ger_tok, eng_tok, USE_BPE):

  if USE_BPE == False:
    ger_voc = build_vocab(config.TRAIN_SRC, ger_tok) #vocab from training data
    eng_voc = build_vocab(config.TRAIN_TGT, eng_tok) #vocab from training data

    train_data = data_process(config.TRAIN_SRC, config.TRAIN_TGT, ger_voc, eng_voc, ger_tok, eng_tok)
    # exit()
    val_data = data_process(config.VAL_SRC, config.VAL_TGT, ger_voc, eng_voc, ger_tok, eng_tok)
    test_data = data_process(config.TEST_SRC, config.TEST_TGT, ger_voc, eng_voc, ger_tok, eng_tok)

    print("train_data ", len(train_data))
    print("valid_data ", len(val_data))
    print("test_data ", len(test_data))

    return ger_voc, eng_voc, train_data, val_data, test_data


def generate_batch(data_batch):
  PAD_IDX = 1 #german_vocab['<pad>']
  SOS_IDX = 2 #german_vocab['<sos>']
  EOS_IDX = 3 #german_vocab['<eos>']

  de_batch, en_batch = [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([SOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([SOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
  de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
  return de_batch, en_batch


def Batcher(train_data, val_data, test_data):
  BATCH_SIZE = 32
  train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=generate_batch)
  valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=generate_batch)
  test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                         shuffle=True, collate_fn=generate_batch)
  return train_iter, valid_iter, test_iter


print("here<<<<<<<<<<<<<<<<<<<<<<")
# exit()