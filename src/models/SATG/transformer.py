import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=600):
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


class TransformerModel(nn.Module):

    def __init__(self, in_size, out_size=64, n_unit=256, n_head=2, n_hid=256, n_layer=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.model_type = 'Transformer'
        self.in_size = in_size
        self.out_size = out_size
        self.n_unit = n_unit
        self.n_head = n_head
        self.n_layers = n_layer
        
        self.src_mask = None
        self.mem = None
        self.mem_ = None
        
        self.encoder = nn.Linear(in_size, n_unit)
        self.pos_encoder = PositionalEncoding(n_unit, dropout)
        encoder_layers = TransformerEncoderLayer(n_unit, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layer)
        self.decoder = nn.Linear(n_unit, out_size)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz, N, M=0):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        for i in range(sz):
            if i>N:
                mask[i][i-N:i+M] = True
                mask[i][:i-N] = False
            else:
                mask[i][:i+M] = True
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         tmp = (torch.triu(torch.ones(sz-N, sz-N)) == 1)
#         mask[N:, :sz-N] = tmp
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, N=100, mem_flg=False):
        src = self.encoder(src)
        # src: (B, T, E)
        src_len = src.size(1)
        if (self.mem is not None) and mem_flg:
            src_ = torch.cat([self.mem, src], dim=1)
        else:
            src_ = src
            
        self.mem = src.detach()
            
        if self.src_mask is None or self.src_mask.size(0) != src_.size(1):
            device = src.device
            mask = self._generate_square_subsequent_mask(src_.size(1), N).to(device)
            self.src_mask = mask
          
        # src: (T, B, E)
        src_ = src_.transpose(0, 1)
        # src: (T, B, E)
        src_ = self.pos_encoder(src_)
        # output: (T, B, E)
        output = self.transformer_encoder(src_, self.src_mask)
        output = output[-src_len:]
        # output: (B, T, E)
        output = output.transpose(0, 1)
        # output: (B, T, C)
        output = self.decoder(output)
        
        return output
    
    def forward_(self, src, N=100, mem_flg=False):
        src_ = self.encoder(src)
        # src: (B, T, E)
        src_len = src_.size(1)
#         if (self.mem_ is not None) and mem_flg:
#             src_ = torch.cat([self.mem_, src], dim=1)
#         else:
#             src_ = src
            
#         self.mem_ = src.detach()
            
        if self.src_mask is None or self.src_mask.size(0) != src_.size(1):
            device = src.device
            mask = self._generate_square_subsequent_mask(src_.size(1), N).to(device)
            self.src_mask = mask
          
        # src: (T, B, E)
        src_ = src_.transpose(0, 1)
        # src: (T, B, E)
        src_ = self.pos_encoder(src_)
        # output: (T, B, E)
        output = self.transformer_encoder(src_, self.src_mask)
        output = output[-src_len:]
        # output: (B, T, E)
        output = output.transpose(0, 1)
        # output: (B, T, C)
        output = self.decoder(output)
        
        return output
    
    def get_attention_weight(self, src, N=100, M=0, mem_flg=True):
        # NOTE: NOT IMPLEMENTED CORRECTLY!!!
        attn_weight = []
        def hook(module, input, output):
            # attn_output, attn_output_weights = multihead_attn(query, key, value)
            # output[1] are the attention weights
            attn_weight.append(output[1].detach().cpu().numpy())
            
        handles = []
        for l in range(self.n_layers):
            handles.append(self.transformer_encoder.layers[l].self_attn.register_forward_hook(hook))

        self.eval()
        with torch.no_grad():
            self.forward_(src, N, mem_flg=mem_flg)

        for handle in handles:
            handle.remove()
#         self.train()

        return attn_weight
#         return torch.stack(attn_weight)
    