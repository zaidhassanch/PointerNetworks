import random

# random.seed(2)
#is this sentence difficult definitely a

def randomizeSentence(sentence):
    augmentedSentence = []
    count = 0


    for word in sentence:
        augmentedSentence.append([word, count])
        count += 1
    # print(augmentedSentence)
    random.shuffle(augmentedSentence)

    count = 0
    for word in augmentedSentence:
        word.append(count)
        count += 1

    return augmentedSentence

sent = ["this", "is","definitely", "a","difficult", "sentence",  ]
print(sent)
s2 = randomizeSentence(sent)
print(s2)

