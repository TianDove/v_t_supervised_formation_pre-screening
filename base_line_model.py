import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy


class Dense_Relu(nn.Module):
    """"""
    def __init__(self, in_dim, out_dim, dropout):
        """"""
        super(Dense_Relu, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        """"""
        res = self.dropout(x)
        res = self.linear(res)
        res = self.act(res)
        return res


class Conv_Bn_Relu(nn.Module):
    """"""

    def __init__(self, in_ch, out_ch, k_size):
        """
        (N, C, L)
        """
        super(Conv_Bn_Relu, self).__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, k_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.conv(x)
        res = self.bn(res)
        res = self.relu(res)
        return res



class ResidualBlock(nn.Module):
    """
    (N, C, L)
    """
    def __init__(self, in_ch, out_ch, k_size):
        """"""
        super(ResidualBlock, self).__init__()
        self.model_name = self.__class__.__name__
        self.conv1 = Conv_Bn_Relu(in_ch, out_ch, k_size)
        self.conv2 = Conv_Bn_Relu(out_ch, out_ch, k_size)
        self.conv3 = Conv_Bn_Relu(out_ch, out_ch, k_size)
        self.x_conv = Conv_Bn_Relu(in_ch, out_ch, 1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        res = res + self.x_conv(x)
        return res


class BaseLine_MLP(nn.Module):
    """"""
    def __init__(self, in_dim):
        """"""
        super(BaseLine_MLP, self).__init__()
        self.model_name = self.__class__.__name__
        # dense
        self.dense1 = Dense_Relu(in_dim, 500, 0.1)
        self.dense2 = Dense_Relu(500, 500, 0.2)
        self.dense3 = Dense_Relu(500, 500, 0.2)

        # last layer
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(500, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """"""
        res = self.dense1(x)
        res = self.dense2(res)
        res = self.dense3(res)
        res = self.dropout(res)
        res = self.linear(res)
        res = self.softmax(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model


class BaseLine_FCN(nn.Module):
    """"""
    def __init__(self, in_dim):
        """"""
        super(BaseLine_FCN, self).__init__()
        self.model_name = self.__class__.__name__
        # conv layer
        self.conv1 = Conv_Bn_Relu(1, 128, 7)
        self.conv2 = Conv_Bn_Relu(128, 256, 5)
        self.conv3 = Conv_Bn_Relu(256, 128, 3)

        # last layer
        self.linear = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """"""
        x = x.unsqueeze(dim=1)
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        # global pooling
        res = F.adaptive_avg_pool1d(res, 1)
        res = res.squeeze()
        # out layer
        res = self.linear(res)
        res = self.softmax(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model


class BaseLine_ResNet(nn.Module):
    """"""
    def __init__(self, in_dim):
        """"""
        super(BaseLine_ResNet, self).__init__()
        self.model_name = self.__class__.__name__
        self.residual1 = ResidualBlock(1, 64, 7)
        self.residual2 = ResidualBlock(64, 128, 5)
        self.residual3 = ResidualBlock(128, 128, 3)
        # out layer
        self.linear = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        """"""
        x = x.unsqueeze(dim=1)
        res = self.residual1(x)
        res = self.residual2(res)
        res = self.residual3(res)
        # global pooling
        res = F.adaptive_avg_pool1d(res, 1)
        res = res.squeeze()
        # out layer
        res = self.linear(res)
        res = self.softmax(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model
