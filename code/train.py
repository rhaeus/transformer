#!/usr/bin/python3

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

import io
import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from datetime import datetime
import os

sents = []
scors = []

from transformer_model import PositionalEncoding, TransformerModel

class PaperOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        
        self._step += 1
        # print("step: ", self._step)
        rate = self.rate()
        # print("rate: ", rate)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step_num = None):
        "Implement `lrate` above"
        if step_num is None:
            step_num = self._step

        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step_num ** (-0.5), step_num * self.warmup ** (-1.5)))

    def get_last_lr(self):
      return self._rate

    def zero_grad(self):
      self.optimizer.zero_grad()

class Trainer:
    def __init__(self):
        file_name = "training_log.txt"
        if os.path.exists(file_name):
          os.remove(file_name)
        self.log_file = open(file_name, "a")

        self.stepcount = 0

        self.batch_size = 20
        self.eval_batch_size = 10
        self.bptt = 35
        self.emsize = 128 # embedding dimension d_model
        self.nhid = 2048 # the dimension of the feedforward network model in nn.TransformerEncoder
        self.nlayers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.nhead = 2 # the number of heads in the multiheadattention models
        self.dropout = 0.15 # the dropout value
        self.epochs = 12 # The number of epochs

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("using device: ", self.device)
        
        self.load_data()

        # (self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        self.model = TransformerModel(self.ntokens, self.emsize, self.nhead, self.nhid, self.nlayers, self.dropout).to(
            self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = 1.5  # learning rate
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)  # try Adam
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

        # optim = torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)  # try Adam
        self.optimizer = PaperOpt(self.emsize, 1200, 4000, optim)

    def load_data(self):
        print("loading data..")
        # build vocab
        # self.train_iter = WikiText2(split='train')
        with open('all_hp.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.tokenizer = get_tokenizer('basic_english')
        counter = Counter()
        for line in lines:
            counter.update(self.tokenizer(line))
        self.vocab = Vocab(counter)

        self.ntokens = len(self.vocab.stoi)  # the size of vocabulary
        # print(ntokens)

        # load data
        # self.train_iter, self.val_iter, self.test_iter = WikiText2()
        train_len = math.floor(0.8 * len(lines))
        val_len = math.floor(0.9 * len(lines))

        self.train_iter = lines[:train_len]
        self.val_iter = lines[train_len:val_len]
        self.test_iter = lines[val_len:]

        self.train_data = self.data_process(self.train_iter)
        self.val_data = self.data_process(self.val_iter)
        self.test_data = self.data_process(self.test_iter)

        self.train_data = self.batchify(self.train_data, self.batch_size)
        self.val_data = self.batchify(self.val_data, self.eval_batch_size)
        self.test_data = self.batchify(self.test_data, self.eval_batch_size)

        print("data loading done!")

        # train_iter = WikiText2(split='train')
        # tokenizer = get_tokenizer('basic_english')
        # with open('all_hp.txt', 'r', encoding='utf-8') as f:
        #     lines = f.readlines()

        # # tokenizer = get_tokenizer('basic_english')
        # counter = Counter()
        # for line in lines:
        #     counter.update(tokenizer(line))
        #     #print(counter)
        #     #print(line)

        # vocab = Vocab(counter)

    def data_process(self, raw_text_iter):
        data = [torch.tensor([self.vocab[token] for token in self.tokenizer(item)],
                             dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data, bsz):
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(self.device)

    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].reshape(-1)
        return data, target

    def train_epoch(self, epoch):
        self.model.train()  # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        src_mask = self.model.generate_square_subsequent_mask(self.bptt).to(self.device)
        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.bptt)):
            data, targets = self.get_batch(self.train_data, i)
            self.optimizer.zero_grad()
            if data.size(0) != self.bptt:
                src_mask = self.model.generate_square_subsequent_mask(data.size(0)).to(self.device)
            output = self.model(data, src_mask)
            loss = self.criterion(output.view(-1, self.ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.stepcount += 1
            

            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                log = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(self.train_data) // self.bptt, self.optimizer.get_last_lr(),
                                    elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss))
                log += '\n'

                print(log)
                self.log_file.write(log)

                total_loss = 0
                start_time = time.time()

    def evaluate(self, eval_model, data_source):
        eval_model.eval()  # Turn on the evaluation mode
        total_loss = 0.
        src_mask = self.model.generate_square_subsequent_mask(self.bptt).to(self.device)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, self.bptt):
                data, targets = self.get_batch(data_source, i)
                if data.size(0) != self.bptt:
                    src_mask = self.model.generate_square_subsequent_mask(data.size(0)).to(self.device)
                output = eval_model(data, src_mask)
                output_flat = output.view(-1, self.ntokens)
                total_loss += len(data) * self.criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)

    def train(self):
        print("starting training")
        best_val_loss = float("inf")
        
        best_model = None
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
        self.model.to(self.device)

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.train_epoch(epoch)
            val_loss = self.evaluate(self.model, self.val_data)
            log = '-' * 89
            log += '\n'
            log += '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss))
            log += '\n'
            log += '-' * 89
            log += '\n'

            print(log)
            print("steps:", trainer.stepcount)
            self.log_file.write(log)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model

            # self.scheduler.step()

        # time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # model_name = "model_{}.pth".format(time_string)
        torch.save(best_model, "model.pth")
        self.model = best_model

        test_loss = self.evaluate(best_model, self.test_data)
        log = '=' * 89
        log += '\n'
        log += '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss))
        log += '\n'
        log += '=' * 89
        log += '\n'

        print(log)
        self.log_file.write(log)



if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    print("steps:", trainer.stepcount)

    









