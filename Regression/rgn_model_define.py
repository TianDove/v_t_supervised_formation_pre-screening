#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
import sys

import torch
import math
import torch.nn as nn

# define Embedding(Tokenize)


# define Position Encoding
class PositionalEncoding_Fixed(nn.Module):
    """"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """"""
        super(PositionalEncoding_Fixed, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        if d_model > 2:
            pe[:, 0::2] = torch.sin(position * div_term)  # [:, 0::2]，[axis=0所有的数据，axis=2从0开始取值，间隔为2]
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.sin(position * div_term)  # [:, 0::2]，[axis=0所有的数据，axis=2从0开始取值，间隔为2]
            # pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """"""
        x = x + self.pe[:, x.size(1), :]
        return self.dropout(x)


class PositionalEncoding_Learnable(nn.Module):
    """"""
    def __init__(self):
        """"""
        super(PositionalEncoding_Learnable, self).__init__()
        pass

    def forward(self, X):
        pass


# define Encoder
class Encoder_TransformerEncoder(nn.Module):
    """"""
    def __init__(self, d_model, nhd, nly=6, dropout=0.1, hid=2048):
        """"""
        super(Encoder_TransformerEncoder, self).__init__()
        encoder = nn.TransformerEncoderLayer(d_model, nhead=nhd, dim_feedforward=hid, dropout=dropout)
        self.encoder_lays = nn.TransformerEncoder(encoder, nly, norm=None)

    def forward(self, X):
        """"""
        res = self.encoder_lays(X)
        return res


# define Decoder
class Decoder_MLP_Linear(nn.Module):
    """"""
    def __init__(self, d_model, dropout=0.1):
        """"""
        super(Decoder_MLP_Linear, self).__init__()
        linear = nn.Linear
        relu = nn.ReLU
        self.hidden1 = nn.Sequential(linear(d_model, d_model // 2), relu())
        self.dropout2 = nn.Dropout(p=dropout)
        self.hidden2 = nn.Sequential(linear(d_model // 2, d_model // 4), relu())
        self.dropout3 = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_model // 4, 1)

    def forward(self, X):
        """"""
        res = self.hidden1(X)
        res = self.dropout2(res)
        res = self.hidden2(res)
        res = self.dropout3(res)
        res = self.linear(res)
        return res


class Decoder_Conv_Pooling(nn.Module):
    """"""
    def __init__(self, d_model, n_token, nhid, dropout=0.1):
        """"""
        super(Decoder_Conv_Pooling, self).__init__()
        assert (d_model % nhid) == 0
        hid_dim = int(d_model / nhid)
        #
        conv1_1 = nn.Conv2d(1, 32, (2, d_model), stride=1)
        conv1_2 = nn.Conv2d(32, 1, (1, 1), stride=1)

        conv2_1 = nn.Conv2d(1, 32, (n_token, 2 * hid_dim), stride=hid_dim)
        conv2_2 = nn.Conv2d(32, 1, (1, 1), stride=1)

        relu = nn.ReLU()
        pool = nn.AvgPool2d
        dropout = nn.Dropout()

        #
        self.cov_lay_1 = nn.Sequential(conv1_1, relu, dropout, conv1_2, relu, pool((3, 1)))
        self.cov_lay_2 = nn.Sequential(conv2_1, relu, dropout, conv2_2, relu, pool((1, 3)))
        self.last_lay = nn.Sequential(nn.Linear(2, 1), relu)

    def forward(self, X):
        """"""
        x = X.unsqueeze(dim=1)
        res1 = self.cov_lay_1(x)
        res2 = self.cov_lay_2(x)
        res_1_ = res1.squeeze(-1).squeeze(-1)
        res_2_ = res2.squeeze(-1).squeeze(-1)
        res = torch.cat((res_1_, res_2_), dim=1)
        res = self.last_lay(res)
        return res

# define Model
class PE_fixed_EC_transformer_DC_mlp_linear(nn.Module):
    """"""

    def __init__(self, d_model, flatten_len, nhd=8, nly=6, dropout=0.1, hid=2048):
        """"""
        self.model_name = self.__class__.__name__
        super(PE_fixed_EC_transformer_DC_mlp_linear, self).__init__()
        # position encoding
        self.position_encoding = PositionalEncoding_Fixed(d_model, dropout=dropout)
        # encoder
        self.encoder = Encoder_TransformerEncoder(d_model, nhd=nhd, nly=nly, dropout=dropout, hid=hid)
        # flatten
        self.flt = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout))
        # decoder
        self.decoder = Decoder_MLP_Linear(flatten_len, dropout)

    def forward(self, X):
        """"""
        res = self.position_encoding(X)
        res = self.encoder(res)
        res = self.flt(res)
        res = self.decoder(res)
        return res


class PE_fixed_EC_transformer_DC_conv_pooling(nn.Module):
    """"""
    def __init__(self, d_model: int, n_token: int, nhd: int = 8, nly: int = 6, dropout: float = 0.1, hid: int = 2048):
        """"""
        self.model_name = self.__class__.__name__
        super(PE_fixed_EC_transformer_DC_conv_pooling, self).__init__()
        # position encoding
        self.position_encoding = PositionalEncoding_Fixed(d_model, dropout=dropout)
        # encoder
        self.encoder = Encoder_TransformerEncoder(d_model, nhd=nhd, nly=nly, dropout=dropout, hid=hid)
        # decoder
        self.decoder = nn.Sequential(nn.Dropout(p=dropout), Decoder_Conv_Pooling(d_model, n_token, nhd))

    def forward(self, X):
        """"""
        res = self.position_encoding(X)
        res = self.encoder(res)
        res = self.decoder(res)
        return res


class PureMLP(nn.Module):
    """"""
    def __init__(self, in_dim: int = None) -> None:
        """"""
        self.model_name = self.__class__.__name__
        super(PureMLP, self).__init__()
        self.in_dim = in_dim
        # model define
        self.linear1 = nn.Linear(in_dim, in_dim // 2)
        self.act1 = nn.Tanh()
        self.linear2 = nn.Linear(in_dim // 2, in_dim // 4)
        self.act2 = nn.Tanh()
        self.linear3 = nn.Linear(in_dim // 4, 1)

    def forward(self, x: torch.tensor) -> None:
        """"""
        res = self.linear1(x)
        res = self.act1(res)
        res = self.linear2(res)
        res = self.act2(res)
        res = self.linear3(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model


if __name__ == '__main__':
    pass
