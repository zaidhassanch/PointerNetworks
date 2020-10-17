import config
import torch


seq_len = torch.randint(8, 11, (1,))
high = 10

rand_seq = torch.randint(0, high, (1,5))

print(rand_seq)

rand_sorted_ind = torch.argsort(rand_seq, dim=1)

print(rand_sorted_ind)

print(rand_sorted_ind.gather(rand_seq, 1))


