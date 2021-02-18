from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torchtext.data import Field, BucketIterator
import torch
import torch.nn as nn
import torch.optim as optim 
import time
import math
from utils import translate_sentence_lstm
from dataloader import Batcher

num_epochs = 10000
learning_rate = 3e-4

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def printSentences(tokens, lang):
    print()
    size = tokens.shape
    for j in range(size[1]):
        print("j = ", j, end=">> ")
        for i in range(size[0]):
            print(lang.itos[tokens[i][j]], end=" ")
        print()
    print()
 
def train(model, device, load_model, save_model, 
    german_vocab, english_vocab, train_data, valid_data, test_data, batch_size, LOAD_NEW_METHOD):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    # sentence = "ein pferd geht unter einer brücke neben einem boot."
    sentence = "A horse are walking beside a boat under a bridge."
    # sentence = 'a little girl climbing into a wooden playhouse.'
    # sentence = "is man lion a stuffed A at smiling."

    # sentence = ['ein', 'pferd', 'geht', 'unter', 'einer', 'brücke', 'neben', 'einem', 'boot', '.']
    #sentence = ['The', 'study’s', 'questions', 'are', 'carefully', 'worded', 'and', 'chosen', '.']
    # sentence = 'The study questions are carefully worded and chosen.'

    # sentence = ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    pad_idx = english_vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if LOAD_NEW_METHOD:
        train_iterator, valid_iterator, test_iterator = Batcher(train_data, valid_data, test_data)
    else:
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=batch_size,
            # sort_within_batch=True,
            # shuffle=True,
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
        #print(f"Translated example sentence: \n {translated_sentence}")
        for word in translated_sentence:
            print(word, end=" ")
        # running on entire test data takes a while
        print("here1")
        score = bleu(train_data[1:10], model, german_vocab, english_vocab, device, LOAD_NEW_METHOD)
        print(f"Train Bleu score {score * 100:.2f}")
        #
        print("here2")
        score = bleu(test_data[1:10], model, german_vocab, english_vocab, device, LOAD_NEW_METHOD)
        print(f"Test Bleu score {score * 100:.2f}")

        model.train()
        losses = []
        t1 = time.time()
        for batch_idx, batch in enumerate(train_iterator):
            # Get input and targets and get to cuda
            if LOAD_NEW_METHOD:
                inp_data = batch[0].to(device)
                target = batch[1].to(device)
            else:
                inp_data = batch.src.to(device)
                target = batch.trg.to(device)
            # continue
            # Forward prop
            # print(target)
            # printSentences(inp_data, german_vocab)
            # printSentences(target, english_vocab)
            trg = target[:-1, :]
            # print(trg.shape)
            output = model(inp_data, trg)

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
            if batch_idx%100 == 0:
                print("batch " +str(batch_idx))
        t2 = time.time()
        print("epoch time = ", t2 - t1)
        mean_loss = sum(losses) / len(losses)
        scheduler.step(mean_loss)


def train2(model, iterator, optimizer, criterion, clip, LOAD_NEW_METHOD, device):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        if LOAD_NEW_METHOD:
            src = batch[0].to(device)
            target = batch[1].to(device)
        else:
            src = batch.src
            target = batch.trg

        optimizer.zero_grad()

        NEW_TRAIN = True

        if NEW_TRAIN:
            trg = target
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = output.reshape(-1, output.shape[2])

            trg = trg[1:].view(-1)
            loss = criterion(output, trg)

        else:
            trg = target[:-1, :]
            output = model(src, trg)
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            loss = criterion(output, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        if(i%100 == 0):
            print(f'\t{i} Train Loss: {loss.item():.3f}')

    return epoch_loss / len(iterator)



def train1(model, device, load_model, save_model, german_vocab, english_vocab,
      train_data, valid_data, test_data, batch_size, LOAD_NEW_METHOD, attn = True):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    if LOAD_NEW_METHOD:
        train_iterator, valid_iterator, test_iterator = Batcher(train_data, valid_data, test_data)
    else:
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
             batch_size = batch_size,
             sort_within_batch = True,
             sort_key = lambda x : len(x.src),
             device = device)

    pad_idx = english_vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

    N_EPOCHS = 100
    CLIP = 1

    best_valid_loss = float('inf')


    for epoch in range(N_EPOCHS):
        src = "ein pferd geht unter einer brücke neben einem boot ."

        translation = translate_sentence_lstm(model, src, german_vocab, english_vocab, device)

        print("SRC: ", src)
        print("PRE: ", translation)

        start_time = time.time()
        train_loss = train2(model, train_iterator, optimizer, criterion, CLIP, LOAD_NEW_METHOD, device)
        #valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

