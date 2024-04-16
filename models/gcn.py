import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from copy import deepcopy
from sklearn.metrics import f1_score
from torch.nn import init
import torch_sparse
from torch_sparse import SparseTensor, matmul

class GCNConv(Module):

    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        if isinstance(adj, SparseTensor):
            x = torch.mm(x, self.weight) 
            x = matmul(adj, x)
        else:
            x = x.reshape(-1, x.shape[-1])
            x = torch.mm(x, self.weight) 
            x = x.reshape(-1,adj.shape[1],x.shape[-1])
            x = adj @ x
            x = x.reshape(-1, x.shape[-1])
        return x + self.bias
        
            

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'


class GCN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, 
                 dropout = 0.5, batch_norm = False, bias=True):

        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.out_channels =out_channels
        self.convs = nn.ModuleList([])

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels, bias=bias))
        else:
            if batch_norm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for i in range(num_layers-2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                if batch_norm:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
    def forward(self, x, adjs):
        if isinstance(adjs, list):
            for i, conv in enumerate(self.convs):
                x = conv(x, adjs[i])
                if i != len(self.convs) - 1:
                    x = self.bns[i](x) if self.batch_norm else x
                    x = F.relu(x)
                    x = F.dropout(x, self.dropout, training=self.training)
        else:
            for i, conv in enumerate(self.convs):
                x = conv(x, adjs)
                if i != len(self.convs) - 1:
                    x = self.bns[i](x) if self.batch_norm else x
                    x = F.relu(x)
                    x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


    

