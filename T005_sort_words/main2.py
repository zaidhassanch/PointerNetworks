
from dataGenerator import generateWords, sortWords, prepareInputForPtrNet, convertAlphabetsToInts, convertIntsToAlphabets

xx = [];
yy = [];

for i in range(1):
    origList = generateWords(2, 5, 5)
    sortedList = sortWords(origList)
    xt, y = prepareInputForPtrNet(origList, sortedList)
    x = convertAlphabetsToInts(xt)
    xa = convertIntsToAlphabets(x)
    xx.append(x)
    yy.append(y)


print(x)
print(y)
print(xa)
print("abc")

print(xx)
print(yy)