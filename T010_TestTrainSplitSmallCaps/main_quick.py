import spacy


nlp = spacy.load("en_core_web_lg")  # make sure to use larger model!
tokens = nlp("dog cat banana")

# print(dir(tokens[0]))
for t in tokens:
        print(t.vector)
# exit()
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))