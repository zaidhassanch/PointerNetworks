import spacy


def filterSentences():
    sentences = [];
    for i in range(0, 20):   #max 20length sentences
        sentences.append([]);
    f = open("eng-fra.txt", "r")
    line = ""
    count = 0
    fileLines = f.readlines()
    print(len(fileLines))
    #exit()
    prevLine = ""
    for line in fileLines:

        # print("      ===============")
        lines = line.split("\t");
        englishLine  = lines[0];
        englishLines = englishLine.split(".")
        if(len(englishLines)>2): continue

        for engLine in englishLines:
            engLine  = engLine.strip()
            if '&' in engLine: continue
            if '"' in engLine: continue
            if ',' in engLine: continue
            if "'" in engLine: continue
            if ":" in engLine: continue
            if "-" in engLine: continue
            if "%" in engLine: continue
            if "?" in engLine: continue
            if "!" in engLine: continue

            if "0" in engLine: continue
            if "1" in engLine: continue
            if "2" in engLine: continue
            if "3" in engLine: continue
            if "4" in engLine: continue
            if "5" in engLine: continue
            if "6" in engLine: continue
            if "7" in engLine: continue
            if "8" in engLine: continue
            if "9" in engLine: continue
            if "cannot" in engLine: continue

            if(engLine == ""): continue
            tempArr = engLine.split(" ")

            if(len(tempArr)>2 and len(tempArr)<13):
                if (engLine == prevLine):
                    continue
                else:
                    prevLine = engLine
                # print(count, engLine)
                sentences[len(tempArr)].append(engLine)
                count += 1
                # if(count == 100): exit()
    return sentences

# sents = getGoodSentences()
# cnt = 0
# for s in sents:
#     print(cnt, len(s))
#     cnt += 1
#print(sents[4])

nlp = spacy.load("en_core_web_sm")  # make sure to use larger model!
sents = filterSentences()

fc = 0
for s in sents:
    f = open("dataGen/sent_"+str(fc)+".txt", "w")
    fc += 1
    nSentences = len(s);
    for i in range(nSentences):
        f.write(s[i]+"\n")
        
    f.close()


