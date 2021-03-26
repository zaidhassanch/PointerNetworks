import torch
import spacy
from torchtext.data.utils import get_tokenizer

from tqdm import tqdm
import csv

# from torchtext.data.metrics import bleu_score
import sys
from nltk.translate.bleu_score import corpus_bleu

from configs import config

spacy_ger = spacy.load("de")


def translate_sentence(model, sentence, german_vocab, english_vocab, device, max_length=50):
    # Load german tokenizer
    # return sentence

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = german_vocab.encode(sentence)
    else:
        print("Arrays are not ready translate_sentence_bpe")
        exit()

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german_vocab.bos_id())
    tokens.append(german_vocab.eos_id())

    # Go through each german token and convert to an index
    text_to_indices = tokens

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english_vocab.bos_id()]
    # print("input = ")
    # for word in sentence:
    #     print(word, end=" ")
    # print()
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess1 = output.argmax(2)
        best_guess =  best_guess1[-1, :].item()
        outputs.append(best_guess)
        # print(english.vocab.itos[best_guess], end=" ")

        if best_guess == english_vocab.eos_id():
            break
    # print()
    translated_sentence = english_vocab.decode(outputs)
    # remove start token
    return translated_sentence

def translate_sentence_tokens(model, tokens, german_vocab, english_vocab, device, max_length=50):


    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german_vocab.bos_id())
    tokens.append(german_vocab.eos_id())

    text_to_indices = tokens

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english_vocab.bos_id()]
    # print("input = ")
    # for word in sentence:
    #     print(word, end=" ")
    # print()
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess1 = output.argmax(2)
        best_guess =  best_guess1[-1, :].item()
        outputs.append(best_guess)
        # print(english.vocab.itos[best_guess], end=" ")

        if best_guess == english_vocab.eos_id():
            break
    # print()
    translated_sentence = english_vocab.decode(outputs)

    # remove start token
    return translated_sentence

def computeBLEU(data, model, german, english, device):

    e_tok = get_tokenizer('spacy', language='en')

    targets = []
    outputs = []

    for example in tqdm(data):
        src = example[0].tolist()
        trg = example[1].tolist()


        prediction = translate_sentence_tokens(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        trg = english.decode(trg)

        prediction = e_tok(prediction)
        trg = e_tok(trg)

        targets.append([trg])
        outputs.append(prediction)

    return corpus_bleu(targets, outputs)


def writeArrToCSV(arr):
    wtr = csv.writer(open('tsv/Our_BleuScores.csv', 'w'), delimiter=',', lineterminator='\n')
    for x in arr: wtr.writerow([x])
