def writeSentArr(filename, arr):
    with open(filename, 'w') as fw:
        for sent in arr:
            fw.write(sent + "\n")
    return


def writeSentArr_error_tsv(fw, arr):
    # with open(filename, 'w') as fw:

    for sent in arr:
        if  sent.find(" are ") == -1: continue
        sent1 = sent +"\t" +sent
        fw.write(sent1 + "\n")
        sent1 = sent.replace(" are ", " is ") + "\t" + sent
        fw.write(sent1 + "\n")
    return

from nltk.tokenize import word_tokenize, sent_tokenize
#from writeArrSentsToFile import writeSentArr, writeSentArr_error_tsv

# def makeParagrahs(textArr):
#     for i, para in enumerate(textArr):
#         print(i, para)
#     nArr = []
#     p = ""
#     for i, para in enumerate(textArr):
#         p += " "+para
#         if(p[-1]=='.'):
#             nArr.append(p)
#             p = ""
#         else:
#             print(".")
#     if(p!=""):
#         nArr.append(p)
#     return nArr

def removeQuotes(textArr):
    k = 0
    # textArr = makeParagrahs(textArr)
    outArr = []
    for i, para in enumerate(textArr):
        if para.count('“') > 1:
            continue
    
        if para.count('“') == 1 and para.count('”') == 1:
            # print("====", para)
            # exit()
            if (para[0]== '“' and para[-1]=='”'):
                para = para[1:-1]
        # print("====", para)
        sentences = sent_tokenize(para)
        # print(i, para)
        #print(len(sentences))

        for j, sent in enumerate(sentences):
            words = word_tokenize(sent)
            if(len(words)<5 or len(words)>30):
                continue
            k+=1
            sent = sent.replace("_", "")
            # print("   ", k, sent)
            outArr.append(sent)
    # print(">>>", len(outArr))
    return outArr


# text = "Hello, Mr. Jacobs, Nice to meet you! When are you coming. I would love to meet."
# text = 'In consequence of an agreement between the sisters, Elizabeth wrote the next morning to their mother, to beg that the carriage might be sent for them in the course of the day. But Mrs. Bennet, who had calculated on her daughters remaining at Netherfield till the following Tuesday, which would exactly finish Jane’s week, could not bring herself to receive them with pleasure before. Her answer, therefore, was not propitious, at least not to Elizabeth’s wishes, for she was impatient to get home. Mrs. Bennet sent them word that they could not possibly have the carriage before Tuesday; and in her postscript it was added, that if Mr. Bingley and his sister pressed them to stay longer, she could spare them very well. Against staying longer, however, Elizabeth was positively resolved—nor did she much expect it would be asked; and fearful, on the contrary, as being considered as intruding themselves needlessly long, she urged Jane to borrow Mr. Bingley’s carriage immediately, and at length it was settled that their original design of leaving Netherfield that morning should be mentioned, and the request made.'

# Open a file: file
#file = open('../../data/NOVELS/testNovel.txt', mode='r')
def cleanFile(filePath, fileName, outFile):
    #print(fileName[-3:])
    if( fileName[-3:] !=  "txt"):
        return
    fullPath = filePath + fileName
    print(fullPath)
    file = open(fullPath, mode='r')

    # read all lines at once
    text = file.read()

    # print(text)

    # close the file
    file.close()

    # print(text)

    #text = "".join([s for s in text.strip().splitlines(True) if s.strip()])
    # print(text)
    #exit()
    textArr = text.split("\n\n")
    a = []
    for st in textArr:
        st = st.replace("\n", " ")
        a.append(st)
    # print(a)
    # for i, p in enumerate(a):
    #     print(i, p)
    # exit()
    textArr = a
    print(len(textArr))
    outArr = removeQuotes(textArr)
    #writeSentArr("sent_"+fileName, outArr)
    writeSentArr_error_tsv(outFile, outArr)

    # print(sent)
    return

# exit()
#
#
#
#
# sentences = sent_tokenize(text)
# words = word_tokenize(text)
#
#
#
# for i, sent in enumerate(sentences):
#     print(i, sent)
# # print (sentences)
# #print (words)
# exit()
#
# text1 = "I'm going to watch a play tonight."
# text2 = "I like to play guitar."
#
# words1 = word_tokenize(text1)
# pos_tags1 = nltk.pos_tag(words1)
#
# words2 = word_tokenize(text2)
# pos_tags2 = nltk.pos_tag(words2)
#
# print (pos_tags1)
# print (pos_tags2)
#
# from nltk.stem import WordNetLemmatizer
#
# wordnet_lemmatizer = WordNetLemmatizer()
#
# print (wordnet_lemmatizer.lemmatize("geese"))
# print (wordnet_lemmatizer.lemmatize("bottles", 'n'))
# print (wordnet_lemmatizer.lemmatize("said", 'v'))
# print (wordnet_lemmatizer.lemmatize("better", 'a'))
# print (wordnet_lemmatizer.lemmatize("quickly", 'r'))
#
# # from nltk.corpus import wordnet as wn
# # present = wn.present_tense(v)
#
# # - remove less than 4 word sentences
# #