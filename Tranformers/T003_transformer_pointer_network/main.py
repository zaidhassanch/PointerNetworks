
from utils import translate_sentence, bleu, load_checkpoint, printSent
import torch
from data import getData
from train import train
from transfomer import Transformer

german, english, train_data, valid_data, test_data = getData()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

# Model hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
src_pad_idx = english.vocab.stoi["<sos>"]
print(src_pad_idx)
print(english.vocab.itos[src_pad_idx])

model = Transformer(device, embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx).to(device)

load_model = True
save_model = True
learning_rate = 3e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# sentence = "ein pferd geht unter einer brücke neben einem boot."
#
# translated_sentence = translate_sentence(
#     model, sentence, german, english, device, max_length=50
# )
sentence1 = ['ein', 'pferd', 'geht', 'unter', 'einer', 'brücke', 'neben', 'einem', 'boot', '.']
translated_sentence = translate_sentence(
    model, sentence1, german, english, device, max_length=50
)
# exit()
# print(f"Translated1 example sentence: \n {sentence}")
# print(f"Translated1 example sentence: \n {translated_sentence}")

# exit()

train(model, device, load_model, save_model, german, english, train_data, valid_data, test_data, batch_size)
# running on entire test data takes a while
score = bleu(train_data[1:100], model, german, english, device)
print(f"Final Train Bleu score {score * 100:.2f}")

score = bleu(test_data[1:100], model, german, english, device)
print(f"Final Test Bleu score {score * 100:.2f}")