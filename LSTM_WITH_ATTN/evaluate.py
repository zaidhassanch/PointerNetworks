import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config
from interface import tensorFromSentence, indexesFromSentence, tensorsFromPair


def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length=config.MAX_LENGTH):
    with torch.no_grad():
        print(sentence)
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=config.device)


        for ei in range(input_length):
            input_encoder = input_tensor[ei]
            output_encoder, encoder_hidden = encoder(input_encoder, encoder_hidden)
            encoder_outputs[ei] += output_encoder[0,0]

        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[config.SOS_token]], device=config.device)

        decoded_words = []

        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(config.MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_atten = decoder(decoder_input \
                                                                , decoder_hidden, encoder_outputs)

            decoder_attentions[di] = decoder_atten.data
            topv, topi = decoder_output.data.topk(1)

            if topi.item()==config.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateAndShowAttention(encoder1, attn_decoder1, input_lang, output_lang, input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_lang, output_lang, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)