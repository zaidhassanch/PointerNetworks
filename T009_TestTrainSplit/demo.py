from pointerNetwork import PointerNetwork


def evaluate(model, sentence):
  """Evaluate after a train epoch"""
  print('Epoch [{}] -- Evaluate'.format(epoch))

  x_val, y_val, text_val = batch(sentenceData, 8)
  out, _ = model(x_val, y_val, teacher_force_ratio=0.)
  out = out.permute(1, 0)

  for i in range(out.size(0)):
    print("=============================================")
    print("yref", y_val[i], out[i], y_val[i] - out[i])

    print("orig", text_val[i])
    v = torch.Tensor.cpu(out[i]).numpy()
    print("[", end="")
    for index in v:
      print(text_val[i][index]+" ", end="")

    print("]")

def demo():
	inputSentence = input("Enter your sentence: ")
	print("============" + p + "===================" )
	print(type(p))

	if config.GPU == True:
		ptrNet = PointerNetwork(config.HIDDEN_SIZE).cuda()
	else:
		ptrNet = PointerNetwork(config.HIDDEN_SIZE)

	evaluate(ptrNet, inputSentence)


demo()