"""multiple transformaiton and multiple propagation"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_sparse import SparseTensor, matmul

class SGC_rich(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, dropout=0.5, num_linear=2, 
                 bias=True, batch_norm=False):

        super(SGC_rich, self).__init__()
        
        self.dropout = dropout
        self.bias = bias
        self.batch_norm = batch_norm
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.lins = nn.ModuleList([])
        if num_linear == 1:
            self.lins.append(GCNLinear(in_channels, out_channels))
        else:
            self.lins.append(GCNLinear(in_channels, hidden_channels))
            if batch_norm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            for i in range(num_linear-2):
                if batch_norm:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(GCNLinear(hidden_channels, hidden_channels))
            self.lins.append(GCNLinear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, adjs):
        for i, layer in enumerate(self.lins):
            x = layer(x)
            if i != len(self.lins) - 1:
                x = self.bns[i](x) if self.batch_norm else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        if isinstance(adjs, SparseTensor):
            for i in range(self.num_layers):
                x = matmul(adjs, x)
        elif isinstance(adjs, list):
            for i in range(self.num_layers):
                x = matmul(adjs[i], x)
        else:
            for i in range(self.num_layers):
                x = adjs @ x
            x = x.reshape(-1, x.shape[2])
        return F.log_softmax(x, dim=1)


class SGC(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=2, dropout=0.5, num_linear=2, 
                 bias=True, batch_norm=False):

        super(SGC, self).__init__()
        
        self.dropout = dropout
        self.bias = bias
        self.batch_norm = batch_norm
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.conv = GCNLinear(in_channels, out_channels)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, adjs):
        weight = self.conv.weight
        bias = self.conv.bias
        x = torch.mm(x, weight)
        if isinstance(adjs, SparseTensor):
            for i in range(self.num_layers):
                x = matmul(adjs, x)
        elif isinstance(adjs, list):
            for i in range(self.num_layers):
                x = matmul(adjs[i], x)
        else:
            for i in range(self.num_layers):
                x = adjs @ x
            x = x.reshape(-1, x.shape[2])
        x = x + bias
        return F.log_softmax(x, dim=1)

class GCNLinear(Module):

    def __init__(self, in_channels, out_channels, with_bias=True):
        super(GCNLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'


