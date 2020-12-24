
from dataGen.dataGenerator import batch, convertToWordsBatch, convertToWordSingle
#from data import batch

print("=========Generating words================")
X, Y = batch(5)
print(X)
print(Y)
print(X.shape)

print(X.numpy().shape)

Xa = convertToWordsBatch(X.numpy())
print("--------------------***************")
print(Xa)

for x in X:
    xv = convertToWordSingle(x)
    print(">>", xv)
