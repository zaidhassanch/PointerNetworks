
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
        if '"' in engLine: continue
        if ',' in engLine: continue
        if "'" in engLine: continue
        if ":" in engLine: continue
        if "-" in engLine: continue
        if "%" in engLine: continue
        if "?" in engLine: continue
        if "!" in engLine: continue

        if(engLine == ""): continue
        tempArr = engLine.split(" ")

        if(len(tempArr)>2 and len(tempArr)<13):
            if (engLine == prevLine):
                continue
            else:
                prevLine = engLine
            print(count, engLine)
            count += 1
            # if(count == 100): exit()