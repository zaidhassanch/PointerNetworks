import torch

mat1 = torch.randn(2, 2, 3)
mat2 = torch.randn(2, 3, 4)
res = torch.bmm(mat1, mat2)
s = res.size()

print(mat1)
print(mat2)
print(res)