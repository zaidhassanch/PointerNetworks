import time

import torch
print(torch.cuda.is_available())
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.profiler import AdvancedProfiler

from models.transformer.model import Model
from data import getData
from dataloader import Batcher
from utils import translate_sentence, translate_sentence_bpe, computeBLEU, writeArrToCSV

import pytorch_lightning as pl

from pytorch_lightning.metrics.functional import accuracy

from configs import config

class grammarTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # change things as required

        self.nepochs = 0
        self.total_time = 0

        self.bleu_scores = []

        embedding_size = 512
        device = "cuda"
        self.prepare_data_own()
        self.model = Model(device, embedding_size, self.src_vocab_size, self.trg_vocab_size, self.src_pad_idx).to(device)

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

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        # results["progress_bar"]["val_acc"] = results["progress_bar"]["train_acc"]
        # del results["progress_bar"]["train_acc"]
        return results

    def validation_epoch_end(self, val_step_outputs):
        # avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        # avg_val_acc = torch.tensor([x["progress_bar"]["val_acc"] for x in val_step_outputs]).mean()
        #
        # pbar = {'avg_val_acc': avg_val_acc}
        print("Translation Sample =================")

        #"An old man trying to get up from a broken chair
        #A man wearing red shirt sitting under a tree
        device = "cuda"

        for sentence in config.sentences:
            if config.USE_BPE == False:
                translated_sentence = translate_sentence(
                    self,
                    sentence, self.german_vocab, self.english_vocab, device, max_length=50
                )
            else:
                translated_sentence = translate_sentence_bpe(
                    self,
                    sentence, self.german_vocab, self.english_vocab, device, max_length=50
                )

            print("Output", translated_sentence)

        # if config.COMPUTE_BLEU == True and self.nepochs == config.MAX_EPOCHS:
        if config.COMPUTE_BLEU == True and self.nepochs > 0:
            bleu_score = computeBLEU(self.test_data, self, self.german_vocab, self.english_vocab, device)
            self.bleu_scores.append(bleu_score)
            print("BLEU score: ", bleu_score)
            if self.nepochs % 1 == 0:
                writeArrToCSV(self.bleu_scores)
        return

    def prepare_data_own(self):
        self.german_vocab, self.english_vocab, self.train_data, self.valid_data, self.test_data = getData(config.LOAD_NEW_METHOD, config.USE_BPE)
        self.train_iterator, self.valid_iterator, self.test_iterator = Batcher(self.train_data, self.valid_data, self.test_data)

        self.src_vocab_size = len(self.german_vocab)
        self.trg_vocab_size = len(self.english_vocab)

        if config.USE_BPE == False:
            self.src_pad_idx = self.english_vocab.stoi["<pad>"]
            self.pad_idx = self.english_vocab.stoi["<pad>"]
        else:
            self.src_pad_idx = self.english_vocab.pad_id()
            self.pad_idx = self.english_vocab.pad_id()

    def setup(self, stage_name):
        # dataset = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
        # self.train_data, self.val_data = random_split(dataset, [55000, 5000])
        pass



    def train_dataloader(self): # defined here to know number of classes
        # Train, Val split
        train_loader = self.train_iterator
        return train_loader

    def val_dataloader(self):
        val_loader = self.valid_iterator
        return val_loader

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



model = grammarTransformer()
profiler = AdvancedProfiler()

start_time = time.time()

if config.GPUS == 1:
    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, gpus=config.GPUS)
    # trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, gpus=config.GPUS, profiler=profiler)
    # trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, gpus=config.GPUS, profiler=True)
elif config.GPUS == 0:
    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS)
else:
    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, gpus=config.GPUS, accelerator="ddp")

# trainer = pl.Trainer(max_epochs=5,gpus=1, precision=16)
trainer.fit(model)

print("Time taken: ", time.time() - start_time)


# encoder-decoder and decoder save: https://scale.com/blog/pytorch-improvements