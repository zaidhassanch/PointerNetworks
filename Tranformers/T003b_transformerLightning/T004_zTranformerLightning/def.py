
import numpy as np
import scipy.io
import time
import torch

k =[[2, 3],[4,5]]
q =[4, 9, 4]

a = torch.zeros(2,3).tolist()

print(a)

scipy.io.savemat('kq1.mat', mdict={'k': k, 'q': q, 't':a})

print("hello")