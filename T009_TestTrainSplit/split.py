

f = open("../data/englishSentences.txt", "r")

lines = f.readlines();
f.close()

length = len(lines)

newLines = [];
for i in range(length):
  # print(i)
  flag = 0
  for j in range(i+1, length):
    line_i = lines[i]
    line_j = lines[j]
    if(lines[i] == lines[j]):
      print("repeated", i,j)
      print(line_i)
      print(line_j)
      flag = 1
  if(flag == 0):
    newLines.append(lines[i])

f = open("../data/englishSentencesNoRepeat.txt", "w")
f.writelines(newLines)
f.close()


