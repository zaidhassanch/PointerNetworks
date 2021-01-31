import torch
import matplotlib.pyplot as plt

from data_process import prepareData
from MTSSModels import EncoderRNN, AttnDecoderRNN
from evaluate import evaluate, evaluateAndShowAttention, evaluateRandomly, showAttention
from train import train, trainIters

import config

hidden_size = 256


def main():

	input_lang, output_lang, pairs = prepareData("eng", "fra");
	encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(config.device)
	attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(config.device)

	trainIters(encoder1, attn_decoder1, input_lang, output_lang, pairs, 75000, print_every=5000)
	torch.save(encoder1.state_dict(), 'encoder4.dict')
	torch.save(attn_decoder1.state_dict(), 'decoder4.dict')
	encoder1.load_state_dict(torch.load('encoder4.dict'))
	attn_decoder1.load_state_dict(torch.load('decoder4.dict'))

	output_words, attentions = evaluate(
		encoder1, attn_decoder1, input_lang, output_lang, "he is good .")
	plt.matshow(attentions.numpy())

	evaluateAndShowAttention(encoder1, attn_decoder1, input_lang, output_lang, "he are a good boy .")

	evaluateAndShowAttention(encoder1, attn_decoder1, input_lang, output_lang, "she are a good girl .")

	evaluateAndShowAttention(encoder1, attn_decoder1, input_lang, output_lang, "they is good boys .")
	evaluateAndShowAttention(encoder1, attn_decoder1, input_lang, output_lang, "are you go play .")

main()

