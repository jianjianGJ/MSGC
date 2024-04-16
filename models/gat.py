

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch_geometric.utils import dense_to_sparse
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

from torch_geometric.nn import GATConv
class GATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, 
                 negative_slope = 0.2, dropout = 0.0,
                 bias = True):
        super(GATConv, self).__init__(node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = Linear(in_channels, out_channels, bias=False)

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))


        self.reset_parameters()
        self.edge_weight = None

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)
    def set_adj(self, rows, cols, batch=None):
        self.rows = rows
        self.cols = cols
        self.batch = batch
        if batch is not None:
            batch_size = batch.max()+1
            self.adj_t = torch.zeros(size=(batch_size, self.n_syn, self.n_syn), device=self.device)
    def forward(self, x, adj_t):
        C = self.out_channels
        x_l = x_r = self.lin(x).view(-1, C)
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)


        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        if isinstance(adj_t, SparseTensor):
            row, col, _ = adj_t.coo()
            edge_index = torch.vstack([row,col])
            out = self.propagate(edge_index, x=(x_l, x_r),
                                 alpha=(alpha_l, alpha_r))
        else:
            if x_l.shape[0] != adj_t.shape[0]*adj_t.shape[1]:
                x_l = x_r = x_r.repeat(adj_t.shape[0],1)
                alpha_l = alpha_l.repeat(adj_t.shape[0])
                alpha_r = alpha_r.repeat(adj_t.shape[0])
            edge_index, _ = dense_to_sparse(adj_t)
            out = self.propagate(edge_index, x=(x_l, x_r),
                                 alpha=(alpha_l, alpha_r))
        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, alpha_j, alpha_i,
                index, ptr,size_i) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class GAT(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, 
                 dropout=0.5):

        super(GAT, self).__init__()

        self.dropout = dropout

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            dropout=dropout)

        self.conv2 = GATConv(
            hidden_channels,
            out_channels,
            dropout=dropout)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
        
    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # print(x.shape, adj.shape)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)
