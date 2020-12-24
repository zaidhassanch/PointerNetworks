import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def computeAttn(K, Q):
    # compute K^T
    Kt = K.permute(0, 2, 1)

    attnVals = torch.matmul(Q, Kt)
    #print("attnVals.shape", attnVals.shape)
    #print(K.shape)

    attnWeights = F.softmax(attnVals, dim=1)
    #print("attnWeights.shape", attnWeights.shape)

    output = torch.matmul(attnWeights, K)
    #print("output.shape", output.shape)
    output = output.squeeze(1)
    #print("output.shape", output.shape)
    return output


def test():
    K = torch.randn(32, 3, 2)
    Q = torch.randn(32, 1, 2)

    print("Q.shape", Q.shape)
    print("K.shape", K.shape)
    V = computeAttn(K, Q)
    print("output.shape", V.shape)

#test()
