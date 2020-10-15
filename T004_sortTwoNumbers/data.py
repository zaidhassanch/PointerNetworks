"""
Generate random data for pointer network
"""
import torch
from torch.utils.data import Dataset
import config


def batch(batch_size, min_len=5, max_len=12):
  array_len = torch.randint(low=min_len, 
                            high=max_len + 1,
                            size=(1,))

  x = torch.randint(high=10, size=(batch_size, array_len, config.NUM_FEATURES))
  return x, torch.sum(x, dim=2).argsort(dim=1)

