import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from seq2seq_attn import  Seq2Seq
from utils import translate_sentence #,calculate_bleu
from data import getData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SRC, TRG, train_data, valid_data, test_data = getData(False)

src_vocab_size = len(SRC)
trg_vocab_size = len(TRG)
SRC_PAD_IDX = SRC.stoi[SRC.pad_token]
TRG_EOS_TOKEN = SRC.stoi[SRC.eos_token]

model = Seq2Seq(SRC_PAD_IDX, src_vocab_size, trg_vocab_size, device, TRG_EOS_TOKEN).to(device)

model.load_state_dict(torch.load('tut4-model.pt'))

src = "ein pferd geht unter einer brücke neben einem boot ."
translation, attention = translate_sentence(model, src, SRC, TRG, device)
print(src)
print(translation)
#exit()
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

example_idx = 12


src = train_data.examples[example_idx].src
trg = train_data.examples[example_idx].trg

print(f'src = {src}')
print(f'trg = {trg}')

# Then we'll use our `translate_sentence` function to get our predicted translation
# and attention. We show this graphically by having the source sentence on the x-axis
# and the predicted translation on the y-axis. The lighter the square at the intersection
# between two words, the more attention the model gave to that source word when
# translating that target word.
#
# Below is an example the model attempted to translate, it gets the translation correct
# except changes *are fighting* to just *fighting*.

translation, attention = translate_sentence(model, src, SRC, TRG, device)

print(f'predicted trg = {translation}')

display_attention(src, translation, attention)

example_idx = 14

src = valid_data.examples[example_idx].src
trg = valid_data.examples[example_idx].trg

print(f'src = {src}')
print(f'trg = {trg}')


translation, attention = translate_sentence(model, src, SRC, TRG, device)
print(f'predicted trg = {translation}')
display_attention(src, translation, attention)

example_idx = 18

src = test_data.examples[example_idx].src
trg = test_data.examples[example_idx].trg

print(f'src = {src}')
print(f'trg = {trg}')

translation, attention = translate_sentence(model, src, SRC, TRG, device)
print(f'predicted trg = {translation}')
display_attention(src, translation, attention)

bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)
print(f'BLEU score = {bleu_score*100:.2f}')

