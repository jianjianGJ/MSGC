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




class SAGEConv(Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels


        # self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        # self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, adj_t):
        if isinstance(adj_t, SparseTensor):
            h = matmul(adj_t, x, reduce='add')
            out = self.lin_r(h)####################
            out += self.lin_r(x[:adj_t.size(0)])
        else:
            h1 = self.lin_r(x)
            h1 = h1.reshape(-1, adj_t.shape[1], h1.shape[1])
            h2 = adj_t @ x.reshape(-1, adj_t.shape[1], x.shape[1])
            h2 = h2.reshape(-1, h2.shape[2])
            h2 = self.lin_r(h2)##############################
            h2 = h2.reshape(-1, adj_t.shape[1], h2.shape[1])
            out = h1 + h2 
            out = out.reshape(-1, out.shape[2])
        return out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class SAGE(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, 
                 dropout = 0.5, batch_norm = False):

        super(SAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.out_channels =out_channels
        self.convs = nn.ModuleList([])

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            if batch_norm:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for i in range(num_layers-2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                if batch_norm:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

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


    

