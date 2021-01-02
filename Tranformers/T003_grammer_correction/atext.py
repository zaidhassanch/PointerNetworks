import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import io
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

train_filepaths = ['.data/train.de', '.data/train.en']
val_filepaths = ['.data/val.de', '.data/val.en']
test_filepaths = ['.data/test_2016_flickr.de', '.data/test_2016_flickr.en']

def build_vocab(filepath, tokenizer):
  counter = Counter()
  with io.open(filepath, encoding="utf8") as f:
    for string_ in f:
      counter.update(tokenizer(string_))

  vocab = Vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'], max_size=10000, min_freq=2);
  vocab.init_token = '<sos>'
  vocab.eos_token = '<eos>'
  return vocab

def data_process(ger_path, eng_path, ger_voc, eng_voc, ger_tok, eng_tok):
  raw_de_iter = iter(io.open(ger_path, encoding="utf8"))
  raw_en_iter = iter(io.open(eng_path, encoding="utf8"))
  data = []
  for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
    de_tensor_ = torch.tensor([ger_voc[token] for token in ger_tok(raw_de)],
                            dtype=torch.long)
    en_tensor_ = torch.tensor([eng_voc[token] for token in eng_tok(raw_en)],
                            dtype=torch.long)
    data.append((de_tensor_, en_tensor_))
  return data

def getData3(ger_tok, eng_tok):

  ger_voc = build_vocab('.data/train.de', ger_tok) #vocab from training data
  eng_voc = build_vocab('.data/train.en', eng_tok) #vocab from training data

  train_data = data_process('.data/train.de', '.data/train.en', ger_voc, eng_voc, ger_tok, eng_tok)
  val_data = data_process('.data/val.de', '.data/val.en', ger_voc, eng_voc, ger_tok, eng_tok)
  test_data = data_process('.data/test_2016_flickr.de', '.data/test_2016_flickr.en', ger_voc, eng_voc, ger_tok, eng_tok)

  return ger_voc, eng_voc, train_data, val_data, test_data

def getData2(ger_tok, eng_tok):

  ger_voc = build_vocab('.data/multi30k/train.de', ger_tok) #vocab from training data
  eng_voc = build_vocab('.data/multi30k/train.en', eng_tok) #vocab from training data

  train_data = data_process('.data/multi30k/train.de', '.data/multi30k/train.en', ger_voc, eng_voc, ger_tok, eng_tok)
  val_data = data_process('.data/multi30k/val.de', '.data/multi30k/val.en', ger_voc, eng_voc, ger_tok, eng_tok)
  test_data = data_process('.data/multi30k/test2016.de', '.data/multi30k/test2016.en', ger_voc, eng_voc, ger_tok, eng_tok)

  print("train_data ", len(train_data))
  print("valid_data ", len(val_data))
  print("test_data ", len(test_data))


  return ger_voc, eng_voc, train_data, val_data, test_data


def generate_batch(data_batch):
  de_batch, en_batch = [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([SOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([SOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
  de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
  return de_batch, en_batch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g_tok = get_tokenizer('spacy', language='de')
e_tok = get_tokenizer('spacy', language='en')
german_vocab, english_vocab, train_data, val_data, test_data = getData2(g_tok, e_tok)

BATCH_SIZE = 32
PAD_IDX = german_vocab['<pad>']
SOS_IDX = german_vocab['<sos>']
EOS_IDX = german_vocab['<eos>']

def Batcher(train_data, val_data, test_data):
  train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=generate_batch)
  valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=generate_batch)
  test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                         shuffle=True, collate_fn=generate_batch)
  return train_iter, valid_iter, test_iter

# for i, (src, trg) in enumerate(train_iter):
#   print(i, src.shape, trg.shape)

print("here<<<<<<<<<<<<<<<<<<<<<<")