import torch
from collections import Counter
from torchtext.vocab import Vocab
import io
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

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

def build_vocab_tsv(filepath, tokenizer, column = 0):
  counter = Counter()
  with io.open(filepath, encoding="utf8") as f:
    for string_ in f:
      string_ = string_.rstrip()
      string_ = string_.split("\t")
      string_ = string_[column]
      counter.update(tokenizer(string_.lower()))

  vocab = Vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'], max_size=40000, min_freq=2);
  vocab.init_token = '<sos>'
  vocab.eos_token = '<eos>'
  vocab.pad_token = '<pad>'
  return vocab

def data_process(ger_path, eng_path, ger_voc, eng_voc, ger_tok, eng_tok):
  SOS_IDX = 2 #german_vocab['<sos>']
  EOS_IDX = 3 #german_vocab['<eos>']
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

    de_tensor_ = torch.cat([torch.tensor([SOS_IDX]), de_tensor_, torch.tensor([EOS_IDX])], dim=0)
    en_tensor_ = torch.cat([torch.tensor([SOS_IDX]), en_tensor_, torch.tensor([EOS_IDX])], dim=0)
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

def data_process_tsv(path, ger_voc, eng_voc, ger_tok, eng_tok):
  SOS_IDX = 2 #german_vocab['<sos>']
  EOS_IDX = 3 #german_vocab['<eos>']
  raw_iter = iter(io.open(path, encoding="utf8"))
  # raw_en_iter = iter(io.open(path, encoding="utf8"))
  data = []
  for raw in raw_iter:
    string_ = raw.lower()
    string_ = string_.rstrip()
    string_ = string_.split("\t")
    raw_de = string_[0]
    raw_en = string_[1]
    de_tensor_ = torch.tensor([ger_voc[token] for token in ger_tok(raw_de)],
                            dtype=torch.long)

    en_tensor_ = torch.tensor([eng_voc[token] for token in eng_tok(raw_en)],
                            dtype=torch.long)

    de_tensor_ = torch.cat([torch.tensor([SOS_IDX]), de_tensor_, torch.tensor([EOS_IDX])], dim=0)
    en_tensor_ = torch.cat([torch.tensor([SOS_IDX]), en_tensor_, torch.tensor([EOS_IDX])], dim=0)
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

def getData2(ger_tok, eng_tok, from_TSV = True):
  if from_TSV == False:
    ger_voc = build_vocab('.data/multi30k/train.de', ger_tok) #vocab from training data
    eng_voc = build_vocab('.data/multi30k/train.en', eng_tok) #vocab from training data

    train_data = data_process('.data/multi30k/train.de', '.data/multi30k/train.en', ger_voc, eng_voc, ger_tok, eng_tok)
    val_data = data_process('.data/multi30k/val.de', '.data/multi30k/val.en', ger_voc, eng_voc, ger_tok, eng_tok)
    test_data = data_process('.data/multi30k/test2016.de', '.data/multi30k/test2016.en', ger_voc, eng_voc, ger_tok, eng_tok)
  else:
    ger_voc = build_vocab_tsv('tsv/combined.tsv', ger_tok, column=0) #vocab from training data
    eng_voc = build_vocab_tsv('tsv/combined.tsv', eng_tok, column=1)

    train_data = data_process_tsv('tsv/combined.tsv', ger_voc, eng_voc, ger_tok, eng_tok)
    #val_data = data_process_tsv('.data/multi30k/val.tsv', ger_voc, eng_voc, ger_tok, eng_tok)
    test_data = data_process_tsv('tsv/combined.tsv', ger_voc, eng_voc, ger_tok, eng_tok)

  print("train_data ", len(train_data))
  # print("valid_data ", len(val_data))
  print("test_data ", len(test_data))

  return ger_voc, eng_voc, train_data, test_data, test_data

def generate_batch(data_batch):

  PAD_IDX = 1 #german_vocab['<pad>']
  de_batch, en_batch = [], []
  # for (de_item, en_item) in data_batch:
  #   de_batch.append(torch.cat([torch.tensor([SOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
  #   en_batch.append(torch.cat([torch.tensor([SOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
  for (de_item, en_item) in data_batch:
    de_batch.append(de_item)
    en_batch.append(en_item)
  de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
  return de_batch, en_batch


def Batcher(train_data, val_data, test_data):
  BATCH_SIZE = 32
  train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=generate_batch)
  valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=generate_batch)
  test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                         shuffle=False, collate_fn=generate_batch)
  return train_iter, valid_iter, test_iter


print("here<<<<<<<<<<<<<<<<<<<<<<")
# exit()