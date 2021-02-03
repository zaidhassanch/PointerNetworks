import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys

spacy_ger = spacy.load("de")

def translate_sentencex2(model, sentence, src_field, trg_field, device, max_len=50):
    model.eval()

    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens.insert(0, src_field.init_token)
    tokens.append(src_field.eos_token)

    src_indexes = [src_field.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    # outputs = [english_vocab.stoi["<sos>"]]

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    mask = model.create_mask(src_tensor)

    trg_indexes = [trg_field.stoi[trg_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.itos[i] for i in trg_indexes]

    return trg_tokens[1:]#, attentions[:len(trg_tokens) - 1]


def translate_sentence(model, sentence, german_vocab, english_vocab, device, max_length=50):

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
            output = model(sentence_tensor, trg_tensor, 0.0)

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


def bleu(data, model, german, english, device, LOAD_NEW_METHOD):
    targets = []
    outputs = []

    for example in data:

        if LOAD_NEW_METHOD:
            src = example[0]
            trg = example[1]
            src = [german.itos[idx] for idx in src]
            trg = [english.itos[idx] for idx in trg]
        else:
            src = example.src
            trg = example.trg

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
