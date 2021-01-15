from utils import translate_sentence, computeBlue, save_checkpoint, load_checkpoint
from torchtext.data import Field, BucketIterator
import torch
import torch.nn as nn
import torch.optim as optim
import time

#from atext import Batcher
# Training hyperparameters

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
            # v = lang.itos[tokens[i][j]]
            v1 = lang.decode(tokens[i][j])
            v2 = lang.decode(token2[i][j])
            
            #if(v == '<sos>' or v == '<pad>' or v == '<eos>' or v == '<unk>'):
            # if(v == lang.bos_id() or v == lang.pad_id() or v == lang.eos_id() or v == lang.unk_id()):
            #     vii = v
            # else:
            #     vi = int(v)+1
            #     if(vi < len(arr[j])):
            #         vii = arr[j][vi]
            #     else:
            #         vii = "<>"
            print(vii, end=" ")
        print()
    print()


def train(learning_rate, model, device, load_model, save_model, german_vocab, english_vocab, train_data, valid_data, test_data, batch_size):
    num_epochs = 10000

    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
        btrain1, btest1 = computeBlue(train_data, test_data, model, german_vocab, english_vocab, device)
        print(f"Train Bleu score_train {btrain1 * 100:.2f},  {btest1 * 100:.2f}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    # pad_idx = english_vocab.stoi["<pad>"]
    pad_idx = english_vocab.pad_id()
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

        
      
        
        model.train()
        losses = []

        for batch_idx, batch in enumerate(train_iterator):
            # Get input and targets and get to cuda
            # print(batch_idx)
            inp_data = batch.src.permute(1,0)
            # print(inp_data.shape)
            
            inp_data = inp_data.to(device)
            target = batch.trg.permute(1,0)
            target = target.to(device)

            #print("inp_data", inp_data.shape)
            #print("target", target.shape)

            # inp_data = batch[0].to(device)
            # target = batch[1].to(device)
            # Forward prop
            # print(target)
            # printSentences(inp_data, german_vocab)
            # printSentences2(target, english_vocab, inp_data, german_vocab)
            trg = target[:-1, :]
            # print(trg.shape)
            #exit()
            optimizer.zero_grad()
            output = model(inp_data, trg, 1)
            # output = model(inp_data, trg, syntax_embedding, arch_flag)

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            

            loss = criterion(output, target)
            losses.append(loss.item())

            if batch_idx % 100 == 0:
                print(batch_idx, " Loss: ", loss)
                end_time = time.time()
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

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        
        btrain1, btest1 = computeBlue(train_data, test_data, model, german_vocab, english_vocab, device)
        
        # load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
        # btrain2, btest2 = computeBlue(train_data, test_data, model, german_vocab, english_vocab, device)
        # print(f"Train Bleu score_train {btrain1 * 100:.2f},  {btest1 * 100:.2f},   {btrain2 * 100:.2f},  {btest2 * 100:.2f}")
        # btrain1, btest1 = computeBlue(train_data, test_data, model, german_vocab, english_vocab, device)
        print(f"Train Bleu score_train {btrain1 * 100:.2f},  {btest1 * 100:.2f}")
