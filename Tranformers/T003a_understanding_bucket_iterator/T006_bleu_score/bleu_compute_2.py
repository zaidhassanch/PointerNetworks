from torchtext.data.metrics import bleu_score

# outputs = [['when', 'you', 'go', 'downhill', ',', 'you', 'have', 'to', 'stick', 'out', 'your', 'chest', 'or', 'you', 'will', 'fall', 'down'], ['i', 'heard', 'a', 'sentence', 'last', 'night', 'while', 'watching', 'tv'], ['when', 'booking', 'an', 'airline', ',', 'we', 'can', 'simply', 'call', 'an', 'agency', 'and', 'do', 'it', 'in', 'a', 'minute']]
# targets = [[['when', 'you', 'go', 'downhill', ',', 'you', 'have', 'to', 'stick', 'out', 'your', 'chest', 'or', 'you', 'will', 'fall', 'down']], [['i', 'heard', 'a', 'sentence', 'last', 'night', 'when', 'i', 'was', 'watching', 'tv']], [['when', 'booking', 'a', 'flight', ',', 'we', 'can', 'just', 'make', 'a', 'phone', 'call', 'to', 'any', 'agency', ',', 'and', 'get', 'it', 'done', 'in', 'one', 'minute']]]

outputs = [['when', 'you', 'go', 'downhill', ',', 'you', 'have', 'to', 'stick', 'out', 'your', 'chest', 'or', 'you', 'will', 'fell', 'down'],['when', 'you', 'go', 'downhill', ',', 'you', 'have', 'to', 'stick', 'out', 'your', 'chest', 'or', 'you', 'will', 'fall', 'down']]
targets = targets = [[['when', 'you', 'go', 'downhill', ',', 'you', 'have', 'to', 'stick', 'out', 'your', 'chest', 'or', 'you', 'will', 'fall', 'down']], [['when', 'you', 'go', 'downhill', ',', 'you', 'have', 'to', 'stick', 'out', 'your', 'chest', 'or', 'you', 'will', 'fall', 'down']]]

bleu1 = bleu_score([outputs[-1]], [targets[-1]])
bleu2 = bleu_score([outputs[-2]], [targets[-2]])


print("BLEU1", bleu1, bleu2, (bleu1+ bleu2)/2)

bleu = bleu_score(outputs, targets)

print("BLEU2", bleu)