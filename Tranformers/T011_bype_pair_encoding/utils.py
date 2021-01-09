import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys

spacy_ger = spacy.load("de")


def translate_sentence(model, sentence, german_vocab, english_vocab, device, max_length=50):
    # Load german tokenizer
    # return sentence

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    # sp_gec.encode

    print("In function translate_sentence")

    if type(sentence) == str:
        tokens = german_vocab.encode(sentence.lower())
        print(tokens)
        # exit()
    else:
        # print(sentence)
        # print(type(sentence))
        # print("Not supported yet array")
        # exit()
        tokens = sentence

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german_vocab.bos_id())
    tokens.append(german_vocab.eos_id())
    # print(tokens)
    # exit()
    # Go through each german token and convert to an index
    # text_to_indices = [german_vocab.stoi[token] for token in tokens]
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
    # print(translated_sentence)
    # print("done")
    # exit()
    #[english_vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence #[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction#[:-1]  # remove <eos> token
        src = german.decode(src)
        trg = english.decode(trg)
        print("src   :>>>", src)
        print("target:   ", trg)
        print("pred  :   ", prediction)
        targets.append([trg])
        outputs.append([prediction])

    # print("calc blue 1")
    # print(outputs)
    # print("====================")
    # print(targets)
    #blue = bleu_score(outputs, targets)
    # blue = bleu_score(targets, targets)
    
    print("calc blue 2")
    return 0


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
