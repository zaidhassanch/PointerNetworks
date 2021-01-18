
from dataGen.dataGenerator import batch, convertToWordsBatch, convertToWordSingle
#from data import batch

print("=========Generating words 1================")
X, Y = batch(4)
# exit()
print(X)
print(Y)
print(X.shape)
# exit()
Xa = convertToWordsBatch(X)
print("--------------------***************")
print(Xa)

for x in X:
    xv = convertToWordSingle(x)
    print(">>", xv)
