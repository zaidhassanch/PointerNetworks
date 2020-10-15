from ptr_net import PointerNetwork, train, evaluate
import time
import torch.optim as optim
import config
from data import batch

EPOCHS = 10
BATCH_SIZE = 32


def main1():

  ptrNet = PointerNetwork(config.HIDDEN_SIZE)
  optimizer = optim.Adam(ptrNet.parameters())
  program_starts = time.time()

  for epoch in range(EPOCHS):
    print('Epoch [{}] -- Train'.format(epoch))

    x, y = batch(BATCH_SIZE)
    train(ptrNet, x, y, optimizer, epoch + 1)

    x_val, y_val = batch(4)
    out, _ = ptrNet(x_val, y_val, teacher_force_ratio=0.)
    print(out.shape)
    out = out.permute(1, 0)
    print(out.shape) 

    sumVal = x_val.sum(dim=2)
    for i in range(out.size(0)):
      print('{} --> {} --> {} --> {}'.format(
        sumVal[i], 
        sumVal[i].gather(0, out[i]),
        sumVal[i].gather(0, y_val[i]),
        sumVal[i].gather(0, out[i]) - sumVal[i].gather(0, y_val[i])
      ))


  now = time.time()
  print("It has been {0} seconds since the loop started".format(now - program_starts))


main1()

# x, y = batch(4)
# print(x.shape)
# # x = x.permute(1,0)
# x0 = x[0]
# print(x0.permute(1,0)) 
# print(y[0])
# s = x0.sum(1);
# print(s)
# print(s.argsort())

# print(x.shape)
# sumVal = x.sum(dim=2)
# print("sumX", sumX.shape)
# for i in range(x.size(0)):
#   print("sumX", sumX[i])
#   print("y", y[i])
#   print('{} --> {}'.format(
#     sumX[i], 
#     sumX[i].gather(0, y[i])
#   ))

  #sumX[i,:,0].gather(0, out[i]),



# print(y)

