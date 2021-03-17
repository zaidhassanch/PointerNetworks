import torch
import spacy

from tqdm import tqdm
import csv

# from torchtext.data.metrics import bleu_score
import sys
from nltk.translate.bleu_score import corpus_bleu

spacy_ger = spacy.load("de")


def translate_sentence(model, sentence, german_vocab, english_vocab, device, max_length=50):
    # Load german tokenizer
    # return sentence

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german_vocab.init_token)
    tokens.append(german_vocab.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german_vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english_vocab.stoi["<sos>"]]
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

        if best_guess == english_vocab.stoi["<eos>"]:
            break
    # print()
    translated_sentence = [english_vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]

def translate_sentence_bpe(model, sentence, german_vocab, english_vocab, device, max_length=50):
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
    tokens.insert(0, german_vocab.stoi[german_vocab.init_token])
    tokens.append(german_vocab.stoi[german_vocab.eos_token])

    text_to_indices = tokens

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english_vocab.stoi["<sos>"]]
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

        if best_guess == english_vocab.stoi["<eos>"]:
            break
    # print()
    translated_sentence = [english_vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]



def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def printSent(arr):
    for item in arr:
        print(item, end=" ")
    print()

def computeBLEU(data, model, german, english, device):
    targets = []
    outputs = []

    for example in tqdm(data):
        src = example[0].tolist()
        trg = example[1].tolist()


        prediction = translate_sentence_tokens(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        trg = [english.itos[idx] for idx in trg]

        targets.append([trg])
        outputs.append(prediction)

    return corpus_bleu(targets, outputs)

def writeArrToCSV(arr):
    wtr = csv.writer(open('tsv/Our_BleuScores.csv', 'w'), delimiter=',', lineterminator='\n')
    for x in arr: wtr.writerow([x])
