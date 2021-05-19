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



class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.ninp = ninp
        self.nhid = nhid

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.layer_norm = nn.LayerNorm(self.nhid)
        # torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers, norm=self.layer_norm)

        
    # self.decoder_layer = nn.TransformerDecoderLayer(d_model=hid_size, nhead = n_head, dim_feedforward=self.pf_size)
    # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layers, norm=self.layer_norm)

        self.encoder = nn.Embedding(ntoken, ninp)

        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask):
        # src.to(device)
        # src_mask.to(device)
        sent_len, batch_s = src.shape[0], src.shape[1]
        # print("sent len ", sent_len)
        # print("batch size ", batch_s)

        # # memory = torch.zeros(1, self.ninp, self.nhid).to(device)
        # memory = torch.zeros(1, batch_s, self.nhid, device=self.device)
        # print("src ", src.size())
        # print("memory ", memory.size())
        # print(mask.size())

        # text embedding and positional encoding
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)




        output = self.transformer_encoder(src, mask)
        # output = self.decoder(src, memory, tgt_mask=mask)
        # output = self.transformer_decoder(src, memory, tgt_mask=mask)
        # output = self.transformer_decoder(src, memory)
        
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Trainer:
    def __init__(self):
        self.batch_size = 20
        self.eval_batch_size = 10
        self.bptt = 35
        self.emsize = 256 # embedding dimension d_model
        self.nhid = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
        self.nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.nhead = 2 # the number of heads in the multiheadattention models
        self.dropout = 0.2 # the dropout value
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("using device: ", self.device)
        self.load_data()

        
        # (self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        self.model = TransformerModel(self.ntokens, self.emsize, self.nhead, self.nhid, self.nlayers, self.dropout).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = 5.0 # learning rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr) #try Adam
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

        

    def load_data(self):
        print("loading data..")
        # build vocab
        # self.train_iter = WikiText2(split='train')
        with open('data/all_hp.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.tokenizer = get_tokenizer('basic_english')
        counter = Counter()
        for line in lines:
            counter.update(self.tokenizer(line))
        self.vocab = Vocab(counter)

        self.ntokens = len(self.vocab.stoi) # the size of vocabulary
        # print(ntokens)

        # load data
        # self.train_iter, self.val_iter, self.test_iter = WikiText2()
        train_len = math.floor(0.8 * len(lines))
        val_len = math.floor(0.9 * len(lines))

        self.train_iter = lines[:train_len]
        self.val_iter= lines[train_len:val_len]
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
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target

    def train_epoch(self, epoch):
        self.model.train() # Turn on the train mode
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

            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(self.train_data) // self.bptt, self.scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def evaluate(self, eval_model, data_source):
        eval_model.eval() # Turn on the evaluation mode
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
        epochs = 5 # The number of epochs
        best_model = None
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
        self.model.to(self.device)

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            self.train_epoch(epoch)
            val_loss = self.evaluate(self.model, self.val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model

            self.scheduler.step()

        time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = "models/model_{}.pth".format(time_string)
        torch.save(best_model.state_dict(), model_name)
        self.model = best_model

        test_loss = self.evaluate(best_model, self.test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)
        return best_model

    def preprocess_text(self, text):
        src = text.split()
        # print(src)
        data = [torch.tensor([self.vocab[token] for token in self.tokenizer(item)],
                            dtype=torch.long) for item in src]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def generate_sequence(self, model, src, len, topk):
        # model.to('cpu')
        #src = [sent_len]
        src = src.unsqueeze(1)
        #src = [sent_len, 1]
        generate_step = 0
        maxLoop = 10
        while generate_step < len:
            src_mask = model.generate_square_subsequent_mask(src.size(0))
            # src.to(device)
            out = model(src, src_mask)
            # print("out 1 ", out.size())
            # print("out 2", out)
            # print("out 3", out[-1, :])
            # #out = [sent_len + 1, 1, vocab_size]
            res, ind = torch.topk(out[-1, :], topk, dim=1) #torch.return_types.topk(
                                                        # values=tensor([[37.3671, 35.0489, 34.6765, 34.2863, 33.8337]], device='cuda:0',
                                                        # grad_fn=<TopkBackward>),
                                                        # indices=tensor([[291, 246, 146,  57, 314]], device='cuda:0'))
            # print("res", res) # res tensor([[37.3671, 35.0489, 34.6765, 34.2863, 33.8337]], device='cuda:0',grad_fn=<TopkBackward>)
            # print("ind", ind) # ind tensor([[291, 246, 146,  57, 314]], device='cuda:0')
            # print("ind size", ind.size(1)) # topk
            perm = torch.randperm(topk)
            idx = perm[:1]
            
            # print("idx", idx)
            # print("idx", idx.item())
            sample = ind[:,idx.item()]

            count = 0
            while sample == 0:
                perm = torch.randperm(topk)
                idx = perm[:1]
                sample = ind[:,idx.item()]
                count += 1
                if count >= 10:
                    break

            # print("sample", sample)
            out = sample.unsqueeze(0)

            # out = torch.argmax(out[-1, :], dim=1) # [1] // index of the largest element: tensor([291], device='cuda:0')
            # print("out1", out)
            # out = out.unsqueeze(0) #[1,1] tensor([[291]], device='cuda:0')
            # print("out2", out)
            src = torch.cat((src, out), dim=0)
            generate_step += 1
            src = src.squeeze(1)
        return src

    def generate_text(self, model):
        # source_sentence = "Cedric Diggory was an extremely handsome boy" 
        # source_sentence = "Hermione came over the crest of the hill"
        source_sentence = "Harry Potter was the"
        # source_sentence = "Why don't you just work you fucking fuck"
        source_sentence = source_sentence.lower()
        # source_sentence = source_sentence.split()
        print(source_sentence)
        model.eval()
        print(' '.join(source_sentence))
        print()
        # x = TEXT.numericalize([source_sentence]).to(device).squeeze(1)
        x = self.preprocess_text(source_sentence)

        # device = "cpu"

        generated_sequence = self.generate_text(x,25,2)
        # print(generated_sequence)
        words = [self.vocab.itos[word_idx] for word_idx in generated_sequence]
        print(' '.join(words))


# def train_model():
#     trainer = Trainer()
#     model = trainer.train()

    
if __name__ == "__main__":
    trainer = Trainer()
    model = trainer.train()
    trainer.generate_text(model)
    

    






