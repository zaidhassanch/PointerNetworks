import torch
import config



def indexesFromSentence(lang, sentence):
    # replace words by indices
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    # convert sentence to index list
    indexes = indexesFromSentence(lang, sentence)
    # end sentence by EOS token always
    indexes.append(config.EOS_token)
    # initialize a torch tensor containing indices
    return torch.tensor(indexes, dtype=torch.long, device=config.device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    # get tensor for input and output respectively
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)