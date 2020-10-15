from ptr_net import Encoder, Decoder, PointerNetwork, train, evaluate
import time
import torch.optim as optim
import config


EPOCHS = 2


encoder = Encoder(config.HIDDEN_SIZE)
decoder = Decoder(config.HIDDEN_SIZE)
ptr_net = PointerNetwork(encoder, decoder)

optimizer = optim.Adam(ptr_net.parameters())

program_starts = time.time()
for epoch in range(EPOCHS):
  train(ptr_net, optimizer, epoch + 1)
  evaluate(ptr_net, epoch + 1)

now = time.time()
print("It has been {0} seconds since the loop started".format(now - program_starts))
