import torch

# tL = 2, b=3, f=4
x = torch.ones(2,3, 4)
print(x)

a = 2 * torch.ones(3,6)

a = torch.unsqueeze(a, 0)
b = a.repeat(2,1,1)

# b = a;
# print(b.shape)
# for i in range(4):
#     print(i)
#     b = torch.cat((b,a),0)

print(b.shape)
# exit()
#b =  torch.empty(1, 3)
print(b.shape, x.shape)
y = torch.cat((x,b), 2)


print(y)
print(y.shape)
# z = torch.cat((x, x, x), 1)
# print(z)
#
# for i in range(5):
#   print(torch.cat((torch.zeros(3, i), torch.zeros(3, 3)), dim=1))
