
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
from seq2seqblvt import  Seq2Seq
from train import train, evaluate, epoch_time
from utils import translate_sentence,calculate_bleu


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

#
SRC = Field(tokenize=tokenize_de, lower=True,
               init_token="<sos>", eos_token="<eos>",  pad_token="<pad>", unk_token="<unk>")

TRG = Field(
    tokenize=tokenize_en, lower=True,
    init_token="<sos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

src_vocab_size = INPUT_DIM
trg_vocab_size = OUTPUT_DIM
model = Seq2Seq(SRC_PAD_IDX, src_vocab_size, trg_vocab_size, device).to(device)

model.load_state_dict(torch.load('tut4-model.pt'))

def display_attention(sentence, translation, attention):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'],
                       rotation=45)
    ax.set_yticklabels(['']+translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


# Now, we'll grab some translations from our dataset and see how well our model did. Note, we're going to cherry pick examples here so it gives us something interesting to look at, but feel free to change the `example_idx` value to look at different examples.
#
# First, we'll get a source and target from our dataset.

# In[25]:


example_idx = 12

src = vars(train_data.examples[example_idx])['src']
trg = vars(train_data.examples[example_idx])['trg']

print(f'src = {src}')
print(f'trg = {trg}')


# Then we'll use our `translate_sentence` function to get our predicted translation and attention. We show this graphically by having the source sentence on the x-axis and the predicted translation on the y-axis. The lighter the square at the intersection between two words, the more attention the model gave to that source word when translating that target word.
#
# Below is an example the model attempted to translate, it gets the translation correct except changes *are fighting* to just *fighting*.

# In[26]:


translation, attention = translate_sentence(src, SRC, TRG, model, device)

print(f'predicted trg = {translation}')


# In[27]:


display_attention(src, translation, attention)


# Translations from the training set could simply be memorized by the model. So it's only fair we look at translations from the validation and testing set too.
#
# Starting with the validation set, let's get an example.

# In[28]:


example_idx = 14

src = vars(valid_data.examples[example_idx])['src']
trg = vars(valid_data.examples[example_idx])['trg']

print(f'src = {src}')
print(f'trg = {trg}')


# Then let's generate our translation and view the attention.
#
# Here, we can see the translation is the same except for swapping *female* with *woman*.

# In[29]:


translation, attention = translate_sentence(src, SRC, TRG, model, device)

print(f'predicted trg = {translation}')

display_attention(src, translation, attention)


# Finally, let's get an example from the test set.

# In[30]:


example_idx = 18

src = vars(test_data.examples[example_idx])['src']
trg = vars(test_data.examples[example_idx])['trg']

print(f'src = {src}')
print(f'trg = {trg}')


# Again, it produces a slightly different translation than target, a more literal version of the source sentence. It swaps *mountain climbing* for *climbing on a mountain*.

# In[31]:


translation, attention = translate_sentence(src, SRC, TRG, model, device)

print(f'predicted trg = {translation}')

display_attention(src, translation, attention)




bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)

print(f'BLEU score = {bleu_score*100:.2f}')

