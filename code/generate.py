#!/usr/bin/python3
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import torch.nn as nn

from transformer_model import PositionalEncoding, TransformerModel

global scors, sents
sents = []
scors = []
class Node:
    def __init__(self, sentence, score, step, src, generator):
        self.children = []
        self.sentence = sentence
        self.score = score
        self.step = step
        self.len = 5
        self.topk = 5
        self.src = src
        self.generator = generator
        # self.tokenizer = get_tokenizer('basic_english')

    def PrintTree(self):
        for i in range(self.topk):
            # print(len(self.children))
            if (self.step == self.len):
                print(self.src)
                print(self.score)
                print(self.step)
                break
            else:
                self.children[i].PrintTree()
        # print(self.sentence)
        # print(self.score)

    def returnTree(self):
        global scors, sents
        for i in range(self.topk):
            if (self.step == self.len):
                sents.append(self.src)
                scors.append(self.score)
                break
                # return self.src, self.score
            else:
                self.children[i].returnTree()

    def buildTree(self):
        if (self.step) < self.len:
            topk1 = 1
            for j in range(1, self.topk + 1):
                # x = Generator.preprocess_text(Generator(), self.src)
                x = self.generator.preprocess_text(self.src)
                # generated_sequence, res, topk1 = Generator.generate_text_beam(Generator(), x, 1, topk1)
                generated_sequence, res, topk1 = self.generator.generate_sequence_beam( x, 1, topk1)
                # words = Generator.returnvocab(Generator(), generated_sequence)
                words = self.generator.returnvocab(generated_sequence)
                # words = [vocab.itos[word_idx] for word_idx in generated_sequence]
                words = ' '.join(words)

                self.children.append(Node(generated_sequence, res + self.score, self.step + 1, words, self.generator))
                self.children[j - 1].buildTree()

class Generator:
    def __init__(self, vocab_text, model_path, vocab_path=None):
      self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
      print("using device: ", self.device)

      self.tokenizer = get_tokenizer('basic_english')

      if vocab_path is not None:
        print('loading vocab..')
        self.vocab = torch.load(vocab_path)
      else:
        print("building vocab..")
        with open(vocab_text, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        
        counter = Counter()
        for line in lines:
            counter.update(self.tokenizer(line))
        self.vocab = Vocab(counter)
        torch.save(self.vocab, "vocab")

      self.ntokens = len(self.vocab.stoi) # the size of vocabulary
      print("vocab has ", self.ntokens, " tokens.")

      print("loading model..")
      self.model = torch.load(model_path).to(self.device)
      self.model.eval()

    def preprocess_text(self, text):
        src = text.split()
        # print(src)
        data = [torch.tensor([self.vocab[token] for token in self.tokenizer(item)],
                            dtype=torch.long) for item in src]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def generate_sequence(self, src, len, topk):
        # model.to('cpu')
        #src = [sent_len]
        src = src.unsqueeze(1).to(self.device)
        #src = [sent_len, 1]
        generate_step = 0
        maxLoop = 10
        while generate_step < len:
            src_mask = self.model.generate_square_subsequent_mask(src.size(0)).to(self.device)
            # src.to(device)
            out = self.model(src, src_mask)
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
            # print("sample ", sample)

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

    def generate_text(self, source_sentence):
      print("generating text..")
      # source_sentence = "Cedric Diggory was an extremely handsome boy" 
      # source_sentence = "Hermione came over the crest of the hill"
      # source_sentence = "Harry Potter was the"
      # source_sentence = "Why don't you just work you fucking fuck"
      source_sentence = source_sentence.lower()
      # source_sentence = source_sentence.split()
      print("Source sentence: ", source_sentence)
      # print(' '.join(source_sentence))
      print()
      # x = TEXT.numericalize([source_sentence]).to(device).squeeze(1)
      x = self.preprocess_text(source_sentence)

      generated_sequence = self.generate_sequence(x,25,2)
      # print(generated_sequence)
      words = [self.vocab.itos[word_idx] for word_idx in generated_sequence]
      print("Generated text: ")
      print(' '.join(words))

    def generate_sequence_beam(self, src, len, topk1):
        # model.to('cpu')
        src = src.unsqueeze(1).to(self.device)
        generate_step = 0

        while generate_step < len:
            topk = topk1
            src_mask = self.model.generate_square_subsequent_mask(src.size(0)).to(self.device)
            out = self.model(src, src_mask)

            m = nn.LogSoftmax(dim=1)
            input = torch.randn(10, 3)
            output = m(out[-1, :])
            res, ind = torch.topk(output, topk, dim=1)
            perm = torch.tensor([topk - 1])

            idx = perm[:1]

            sample = ind[:, idx.item()]
            res = res[:, idx.item()]
            topk += 1

            while sample == 0:
                res, ind = torch.topk(output, topk, dim=1)
                perm = torch.tensor([topk - 1])
                idx = perm[:1]
                sample = ind[:, idx.item()]
                res = res[:, idx.item()]
                topk += 1

            out = sample.unsqueeze(0)

            src = torch.cat((src, out), dim=0)

            generate_step += 1
        src = src.squeeze(1)

        return src, res.item(), topk

    def generate_text_beam(self, source_sentence):
      print("generating text..")
      source_sentence = source_sentence.lower()
      print(source_sentence)
      print()

      x = self.preprocess_text(source_sentence)

      # generated_sequence = self.generate_text(x,25,2)
      len = 60
      # ---
      generated_sequence, res, topk = self.generate_sequence_beam(x, len, 1)
      # print(generated_sequence)
      words = [self.vocab.itos[word_idx] for word_idx in generated_sequence]
      print(' '.join(words))

      nodes = []
      for k in range(int(len / 6)):
          global scors, sents
          scors = []
          sents = []
          nodes.append(Node("start", -20, 0, source_sentence, self))
          nodes[k].buildTree()
          nodes[k].returnTree()
          max = -100000
          index = 0
          a = 0
          for i in (scors):
              if i > max:
                  max = i
                  index = a
              a += 1
          source_sentence = sents[index]
      print(sents[index])

    def returnvocab(self, generated_sequence):
        words = [self.vocab.itos[word_idx] for word_idx in generated_sequence]
        return words

if __name__ == "__main__":
    generator = Generator('all_hp.txt', 'model.pth', 'vocab')
    generator.generate_text_beam("Harry Potter and the")
    generator.generate_text_beam("Harry Potter")
    generator.generate_text_beam("Dumbledore walked through the great hall. The food on the tables looked delicious. Rons ")
    generator.generate_text_beam("What is happening")
    generator.generate_text_beam("Quidditch will be held")
    generator.generate_text_beam("Harry Potter and the")
    generator.generate_text_beam("Quidditch")
    

    