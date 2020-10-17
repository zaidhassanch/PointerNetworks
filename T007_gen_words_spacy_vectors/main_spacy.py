
import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
# nlp = spacy.load("en_core_web_sm")
# import spacy

sentence = "a quick brown fox";


nlp = spacy.load("en_core_web_md")  # make sure to use larger model!
tokens = nlp(sentence)

for t in tokens:
        print(t.vector)
# exit()
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))