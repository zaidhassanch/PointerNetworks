import time
import os
import random
import numpy as np
# import numpy
import torch
print(torch.cuda.is_available())
from torch import nn
from torch import optim
# from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.profiler import AdvancedProfiler

from models.transformer.Model import Model
from data import MyDataModule
from dataloader import Batcher
from utils import translate_sentence, computeBLEU, writeArrToCSV

import pytorch_lightning as pl

from pytorch_lightning.metrics.functional import accuracy
# from models.transformer.multiheadattn import myGlobal
from configs import config

def seed_torch(seed=config.SEED): #move to config
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #seed all gpus
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

class grammarTransformer(pl.LightningModule):
    def __init__(self, german_vocab, english_vocab, test_data):
        super().__init__()
        # change things as required

        self.nepochs = 0
        self.total_time = 0

        self.bleu_scores = []

        embedding_size = 512
        num_heads = config.NUM_HEADS
        num_encoder_layers = config.N_LAYERS
        num_decoder_layers = config.N_LAYERS
        dropout = 0.0
        max_len = config.MAX_LEN
        forward_expansion = config.FORWARD_EXP
        learning_rate = 3e-4
        if config.GPUS==0:
            self.deviceLegacy = "cpu"
        else:
            self.deviceLegacy = "cuda"
        self.german_vocab, self.english_vocab, self.test_data = german_vocab, english_vocab, test_data
        self.src_vocab_size = len(self.german_vocab)
        self.trg_vocab_size = len(self.english_vocab)

        if config.USE_BPE == False:
            self.src_pad_idx = self.english_vocab.stoi["<pad>"]
            self.pad_idx = self.english_vocab.stoi["<pad>"]
        else:
            self.src_pad_idx = self.english_vocab.pad_id()
            self.pad_idx = self.english_vocab.pad_id()

        self.model = Model(
            embedding_size, #512
            self.src_vocab_size, #7855
            self.trg_vocab_size, #5894
            self.src_pad_idx, #1
            num_heads, #1
            num_encoder_layers, #
            num_decoder_layers, #
            forward_expansion, #1
            dropout, #0.0
            max_len, #100
            self.deviceLegacy, #'cuda'
        ).to(self.deviceLegacy)

        # pad_idx = english_vocab.stoi["<pad>"]
        # criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, inp_data, trg):
        output = self.model(inp_data, trg)
        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        inp_data = batch[0]
        target = batch[1]

        # x: b x 1 x 28 x28
        trg = target[:-1, :]

        # 1 forward
        output = self(inp_data, trg)# l: logits

        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        # 2 compute the objective function
        J = self.loss(output, target)
        #       J = torch.autograd.Variable(torch.tensor(0)).to(self.device)


        # acc = accuracy(logits, y)
        # pbar = {'train_acc': acc}

        # return {'loss': J, 'progress_bar': pbar}
        return J

    def setup(self, stage_name):
        # dataset = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
        # self.train_data, self.val_data = random_split(dataset, [55000, 5000])
        pass

    def on_epoch_start(self):
        print(">>>>>>>>>>>>>>>>>>>>> on_epoch_start")
        self.start_time = time.time()
        self.nepochs += 1

    def on_epoch_end(self):
        print(">>>>>>>>>>>>>>>>>>>>> on_epoch_end1")
        epoch_time = time.time() - self.start_time

        self.total_time += epoch_time
        # print(">>>>>>>>>>>>>>>>>>>>> on_epoch_end2", self.nepochs)
        print("Epoch Time taken: ", epoch_time, self.total_time / self.nepochs)
        for sentence in config.sentences:
            # if self.nepochs == config.MAX_EPOCHS:
            #     myGlobal.change(True)
            # myGlobal = True
            translated_sentence = translate_sentence(
                self,
                sentence, self.german_vocab, self.english_vocab, self.deviceLegacy,max_length=50
            )
            # print("Output", translated_sentence)
            # print(sentence)
            # global myGlobal
            # myGlobal = False
            # exit()
            # if self.nepochs == config.MAX_EPOCHS:
            #     myGlobal.change(False)
            #     print("Input", sentence)
            #     print("Output", translated_sentence)
            #     exit()
            print("Output", translated_sentence)

        # if config.COMPUTE_BLEU == True and self.nepochs == config.MAX_EPOCHS:
        if config.COMPUTE_BLEU == True and self.nepochs > 0:
            bleu_score = computeBLEU(self.test_data[1:100], self, self.german_vocab, self.english_vocab, self.deviceLegacy)
            self.bleu_scores.append(bleu_score)
            print("BLEU score: ", bleu_score)
            if self.nepochs % 1 == 0:
                writeArrToCSV(self.bleu_scores)

seed_torch()

dm = MyDataModule()
model = grammarTransformer(dm.german_vocab, dm.english_vocab, dm.test_data)

profiler = AdvancedProfiler()

start_time = time.time()

if config.GPUS == 1:
    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, gpus=config.GPUS, precision=config.PRECISION)
    # trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, gpus=config.GPUS, profiler=profiler)
    # trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, gpus=config.GPUS, profiler=True)
elif config.GPUS == 0:
    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, precision=config.PRECISION)
else:
    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, gpus=config.GPUS, accelerator="ddp", precision=config.PRECISION)

# trainer = pl.Trainer(max_epochs=5,gpus=1, precision=16)
trainer.fit(model, dm)

print("Time taken: ", time.time() - start_time)


# encoder-decoder and decoder save: https://scale.com/blog/pytorch-improvements
