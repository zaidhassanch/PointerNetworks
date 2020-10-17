
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

# =========Generating words================
# tensor([[[8, 6, 3, 1, 1, 9],
#          [4, 9, 4, 5, 5, 2],
#          [5, 7, 9, 2, 1, 6],
#          [9, 8, 0, 4, 7, 8],
#          [4, 2, 6, 7, 4, 0]],
#
#         [[6, 5, 3, 6, 3, 0],
#          [4, 6, 8, 9, 5, 9],
#          [4, 5, 8, 7, 2, 4],
#          [7, 1, 0, 8, 1, 2],
#          [7, 0, 8, 2, 4, 3]]])
# tensor([[4, 0, 1, 2, 3],
#         [3, 0, 4, 2, 1]])