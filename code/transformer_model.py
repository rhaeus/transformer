import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math

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
        # decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        # self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers, norm=self.layer_norm)

        
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
        # tutorial 
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

        # xavier v1
        # for p in self.encoder.parameters():
        #   if p.dim() > 1:
        #       nn.init.xavier_uniform_(p)

        # for p in self.decoder.parameters():
        #   if p.dim() > 1:
        #       nn.init.xavier_uniform_(p)

        # for p in self.transformer_encoder.parameters():
        #   if p.dim() > 1:
        #       nn.init.xavier_uniform_(p)

        # xavier v2
        # self.decoder.bias.data.zero_()
        # nn.init.xavier_uniform_(self.encoder.weight.data)
        # nn.init.xavier_uniform_(self.decoder.weight.data)


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
