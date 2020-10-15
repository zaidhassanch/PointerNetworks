from ptr_net import PointerNetwork, train, evaluate
import time
import torch.optim as optim
import config
from data import batch

EPOCHS = 1
BATCH_SIZE = 32


def main():

  ptrNet = PointerNetwork(config.HIDDEN_SIZE)
  optimizer = optim.Adam(ptrNet.parameters())
  program_starts = time.time()

  for epoch in range(EPOCHS):
    print('Epoch [{}] -- Evaluate'.format(epoch))

    x, y = batch(BATCH_SIZE)
    train(ptrNet, x, y, optimizer, epoch + 1)

    x_val, y_val = batch(4)
    out, _ = ptrNet(x_val, y_val, teacher_force_ratio=0.)

    out = out.permute(1, 0)

    for i in range(out.size(0)):
      print('{} --> {} --> {}'.format(
        x_val[i], 
        x_val[i,:,0].gather(0, out[i]),
        x_val[i,:,0].gather(0, y_val[i])
      ))


  now = time.time()
  print("It has been {0} seconds since the loop started".format(now - program_starts))


main()