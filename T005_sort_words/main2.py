
from dataGenerator import generateWords, sortWords, prepareInputForPtrNet, convertAlphabetsToInts, convertIntsToAlphabets

xx = [];
yy = [];

for i in range(4):
    origList = generateWords(2, 3, 3)
    sortedList = sortWords(origList)
    xt, y = prepareInputForPtrNet(origList, sortedList)
    x = convertAlphabetsToInts(xt)
    xa = convertIntsToAlphabets(x)
    print(x)
    print(y)
    print(xa)

    xx.append(x)
    yy.append(y)


print("=============================================")

print(xx)
print(yy)