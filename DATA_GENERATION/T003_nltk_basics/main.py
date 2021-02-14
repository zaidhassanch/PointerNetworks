from nltk.stem.wordnet import WordNetLemmatizer
words = ['gave','went','going','dating']
for word in words:
    print(word+"-->"+WordNetLemmatizer().lemmatize(word,'v'))