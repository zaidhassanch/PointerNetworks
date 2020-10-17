


def generateWordBatch(nBatchSize):
    x1 = ['bd', 'cc', 'az']

    nList = []
    count = 0
    for word in x1:
        nList.append([word, count])
        count += 1

    return nList

def sortWords(list):
    newList = list.copy()
    newList.sort(key=lambda e: e[0])
    return newList

def prepareInputForPtrNet(origList, sortedList):
    input = [x[0] for x in origList]
    target = [x[1] for x in sortedList]
    return input, target

origList = generateWordBatch(2)
sortedList = sortWords(origList)
x, y = prepareInputForPtrNet(origList, sortedList)

print(x)
print(y)

# Prepare input for our Neural Net
# - N lettered words and their sort order
#
# - Sort words to get sort order
#   > We have sorting logic already
#
# - Convert letters to arrays
# x = [[54, 58], [55, 55], [53, 65]]
# y = [2, 0, 1]
#
# - Convert array to letters BACK
#
# - done