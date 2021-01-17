from torchtext.data.metrics import bleu_score  #ahmad1

outputs = [['several', 'men', 'in', 'hard', 'hats', 'are', 'operating', 'a', 'giant', 'pulley', 'system', '.'], ['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.'], ['a', 'man', 'in', 'a', 'blue', 'shirt', 'is', 'standing', 'on', 'a', 'ladder', 'cleaning', 'a', 'window', '.'], ['two', 'men', 'are', 'at', 'the', 'stove', 'preparing', 'food', '.'], ['a', 'man', 'in', 'green', 'holds', 'his', 'guitar', 'while', 'the', 'other', 'man', 'observes', 'his', 'shirt', '.'], ['a', 'man', 'is', 'smiling', 'at', 'a', 'stuffed', 'lion'], ['a', 'trendy', 'girl', 'talking', 'on', 'her', 'cellphone', 'while', 'gliding', 'slowly', 'down', 'the', 'street', '.'], ['a', 'woman', 'with', 'a', 'large', 'purse', 'is', 'walking', 'by', 'a', 'gate', '.'], ['boys', 'dancing', 'on', 'poles', 'in', 'the', 'middle', 'of', 'the', 'night', '.']]
targets = [[['several', 'men', 'in', 'hard', 'hats', 'are', 'operating', 'a', 'giant', 'pulley', 'system', '.']], [['a', 'little', 'girl', 'climbing', 'into', 'a', 'wooden', 'playhouse', '.']], [['a', 'man', 'in', 'a', 'blue', 'shirt', 'is', 'standing', 'on', 'a', 'ladder', 'cleaning', 'a', 'window', '.']], [['two', 'men', 'are', 'at', 'the', 'stove', 'preparing', 'food', '.']], [['a', 'man', 'in', 'green', 'holds', 'a', 'guitar', 'while', 'the', 'other', 'man', 'observes', 'his', 'shirt', '.']], [['a', 'man', 'is', 'smiling', 'at', 'a', 'stuffed', 'lion']], [['a', 'trendy', 'girl', 'talking', 'on', 'her', 'cellphone', 'while', 'gliding', 'slowly', 'down', 'the', 'street', '.']], [['a', 'woman', 'with', 'a', 'large', 'purse', 'is', 'walking', 'by', 'a', 'gate', '.']], [['boys', 'dancing', 'on', 'poles', 'in', 'the', 'middle', 'of', 'the', 'night', '.']]]



blue = bleu_score(outputs, targets)
print("BLUE SCORE = ", blue)