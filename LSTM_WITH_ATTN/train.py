import torch
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from interface import tensorsFromPair, indexesFromSentence, tensorFromSentence

import config
import time
import random
import math




def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=config.MAX_LENGTH):


    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.initHidden()

    input_length = input_tensor.size()[0]
    target_length = target_tensor.size()[0]

    encoder_outputs = torch.zeros((max_length, encoder.hidden_size), device=config.device)

    loss = 0 # remember to zero loss
    for ei in range(input_length):
        output_encoder, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] += output_encoder[0,0]
    
    decoder_input = torch.tensor([[config.SOS_token]], device=config.device)
    decoder_hidden = encoder_hidden

    teacher_force = 0.7

    for di in range(target_length):
        decoder_output, decoder_hidden, attention_weights = decoder( \
            decoder_input, decoder_hidden, encoder_outputs)

        topv, topi = decoder_output.topk(1)
        loss += criterion(decoder_output, target_tensor[di])

        if  teacher_force< config.teacher_forcing_ratio:
            decoder_input = target_tensor[di]
        else:
            decoder_input = topi.squeeze().detach()
            if decoder_input.item() == config.EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, input_lang, output_lang, pairs, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0 # Reset every print_every

    # optimize over parameters defined in encoder and decoder
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # create index lists for input and output

    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]

    # NLLLoss:The negative log likelihood loss. It is useful to train a classification problem with C classes
    # applied where logloss has been applied
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        # select training pair for this iteration
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # decoder encoder step is taken inside this function
        loss = train(input_tensor, target_tensor, encoder, \
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            plot_loss_total = 0
    showPlot(plot_losses)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()