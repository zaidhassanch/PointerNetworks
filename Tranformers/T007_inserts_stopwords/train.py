from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torchtext.data import Field, BucketIterator
import torch
import torch.nn as nn
import torch.optim as optim 

#from atext import Batcher
# Training hyperparameters
num_epochs = 10000
learning_rate = 3e-4

def printSentences(tokens, lang):
    print()
    size = tokens.shape
    for j in range(size[1]):
        print("j = ", j, end=">> ")
        for i in range(size[0]):
            print(lang.itos[tokens[i][j]], end=" ")
        print()
    print()

def printSentencesx(tokens, lang):
    # print()
    outarr = []
    size = tokens.shape
    for j in range(size[1]):
        # print("j = ", j, end=">> ")
        inarr = []
        for i in range(size[0]):
            inarr.append(lang.itos[tokens[i][j]])
        # print()
        outarr.append(inarr)
    # print()
    return outarr


def printSentences2(tokens, lang, token2, lang2):
    arr = printSentencesx(token2, lang2)
    print(arr)
    # return
    size = tokens.shape
    for j in range(size[1]):
        print("j = ", j, end=">> ")
        for i in range(size[0]):
            v = lang.itos[tokens[i][j]]
            if(v == '<sos>' or v == '<pad>' or v == '<eos>' or v == '<unk>'):
                vii = v
            else:
                vi = int(v)+1
                if(vi < len(arr[j])):
                    vii = arr[j][vi]
                else:
                    vii = "<>"
            print(vii, end=" ")
        print()
    print()


def train(model, device, load_model, save_model, german_vocab, english_vocab, train_data, valid_data, test_data, batch_size):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        load_checkpoint(torch.load("RSWI_checkpoint.pth.tar"), model, optimizer)

    # sentence = "ein pferd geht unter einer brücke neben einem boot."
    # sentence = 'a little girl climbing into a wooden playhouse.'
    sentence = "man stuffed smiling lion"
    #6 1 4 7 3 2 5 0
    # sentence = ['ein', 'pferd', 'geht', 'unter', 'einer', 'brücke', 'neben', 'einem', 'boot', '.']
    #sentence = ['The', 'study’s', 'questions', 'are', 'carefully', 'worded', 'and', 'chosen', '.']
    # sentence = 'The study questions are carefully worded and chosen.'

    # sentence = ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    pad_idx = english_vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # train_iterator, valid_iterator, test_iterator = Batcher(train_data, valid_data, test_data)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device,
        )

    step = 0

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        model.eval()
        # sentence = "Das wird sehr seltsam"
        # sentence = "Frankreich wird wohl Deutschland angreifen"

        translated_sentence = translate_sentence(
            model,
            sentence, german_vocab, english_vocab, device, max_length=50
        )

        print(f"Translated example sentence: \n {sentence}")
        print(f"Translated example sentence: \n {translated_sentence}")
        # exit()

        # running on entire test data takes a while
        print("here1")
        score = bleu(train_data[1:10], model, german_vocab, english_vocab, device)
        print(f"Train Bleu score {score * 100:.2f}")

        print("here2")
        score = bleu(test_data[1:50], model, german_vocab, english_vocab, device)
        print(f"Test Bleu score {score * 100:.2f}")

        model.train()
        losses = []

        for batch_idx, batch in enumerate(train_iterator):
            # Get input and targets and get to cuda
            # print(batch_idx)
            inp_data = batch.src
            inp_data = inp_data.to(device)
            target = batch.trg
            target = target.to(device)

            # inp_data = batch[0].to(device)
            # target = batch[1].to(device)
            # Forward prop
            # print(target)
            # printSentences(inp_data, german_vocab)
            # printSentences2(target, english_vocab, inp_data, german_vocab)
            trg = target[:-1, :]
            # print(trg.shape)
            output = model(inp_data, trg)
            # output = model(inp_data, trg, syntax_embedding, arch_flag)

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output, target)
            losses.append(loss.item())

            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # plot to tensorboard
            # writer.add_scalar("Training loss", loss, global_step=step)
            step += 1

        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)

        
