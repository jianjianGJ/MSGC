#%% Imports
import torch
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from sklearn.cluster import BisectingKMeans 
from torch_scatter import scatter_mean
from collections import Counter
import torch.nn as nn
#%% Samplers
class SamplerForClass:
    '''
    In inductive setting, sample $n_need_max nodes with label $class_id.
    Return nodes ids, type:tensor.
    '''
    def __init__(self, labels, n_classes):
        self.n_nodes = labels.shape[0]
        idx_all = torch.arange(self.n_nodes)
        self.class2idx = {}
        for i in range(n_classes):
            self.class2idx[i] = idx_all[labels == i]
    def sample_from_class(self, class_id, n_need_max=256):
        return self.class2idx[class_id]\
            [torch.randperm(self.class2idx[class_id].shape[0])[:n_need_max]]

class NeighborSamplerForClass:
    def __init__(self, adj_t, labels, n_classes, sizes):
        self.n_nodes = labels.shape[0]
        self.device = adj_t.device()
        idx_all = torch.arange(self.n_nodes)
        self.class2idx = {}
        for i in range(n_classes):
            self.class2idx[i] = idx_all[labels == i]
        self.samplers = []
        for i in range(n_classes):
            nodes_idx = torch.LongTensor(self.class2idx[i])
            self.samplers.append(NeighborSampler(adj_t,
                                    node_idx=nodes_idx,
                                    sizes=sizes, 
                                    num_workers=12, return_e_id=False,
                                    num_nodes=self.n_nodes,
                                    shuffle=True))
    def sample_from_class(self, class_id, batch_size=16):
        batch = self.class2idx[class_id]\
            [torch.randperm(self.class2idx[class_id].shape[0])[:batch_size]]
        batch_size, n_id, adjs = self.samplers[class_id].sample(batch)
        adjs = [item[0].to(self.device) for item in adjs]
        return batch_size, n_id, adjs
#%% Core functions
def assign_each_y_proportionally(y_train, n_syn, n_classes):
    assert n_syn >= n_classes
    device = y_train.device
    y_train = y_train[y_train>=0]
    n = y_train.shape[0]
    base = torch.ones(n_classes, device=device)
    rate = F.one_hot(y_train, num_classes=n_classes).sum(0)/n
    n_each_y = torch.floor((n_syn-base.sum())*rate)+base
    left = n_syn - n_each_y.sum()
    for _ in range(int(left.item())):
        more = n_each_y/n_each_y.sum()/rate
        n_each_y[more.argmin()] += 1
    n_each_y = n_each_y.to(torch.int64)
    
    y_syn = torch.LongTensor(n_syn).to(device)
    start = 0
    starts = torch.zeros_like(n_each_y)
    for c in range(n_classes):
        y_syn[start:start+n_each_y[c]] = c
        starts[c] = start
        start += n_each_y[c]
    return y_syn, n_each_y, starts
def initialize_x_syn(x_train, y_train, y_syn, n_classes, initial):
    if initial=='cluster':
        x_syn = torch.zeros(y_syn.shape[0],x_train.shape[1])
        for c in range(n_classes):
            x_c = x_train[y_train==c].cpu()
            n_c = (y_syn==c).sum().item()
            k_means = BisectingKMeans(n_clusters=n_c, random_state=0)
            k_means.fit(x_c)
            clusters = torch.LongTensor(k_means.predict(x_c))
            x_syn[y_syn==c] = scatter_mean(x_c,clusters, dim=0)
        return x_syn.to(x_train.device)
    elif initial=='sample':
        sam = SamplerForClass(y_train, n_classes)
        counter = Counter(y_syn.cpu().numpy())
        idx_selected_list = []
        for c in range(n_classes):
            idx_c = sam.sample_from_class(c, n_need_max=counter[c])
            idx_selected_list.append(idx_c)
        idx_selected = torch.cat(idx_selected_list).to(x_train.device)
        return x_train[idx_selected]
    elif initial=='mean':
        x_syn = torch.zeros(y_syn.shape[0],x_train.shape[1]).to(x_train.device)
        for c in range(n_classes):
            x_c = x_train[y_train==c]
            n_c = (y_syn==c).sum()
            x_syn[y_syn==c] = x_c.mean(0)
        return x_syn
class GraphSynthesizer(nn.Module):

    def __init__(self, x_train, y_train, adj_t_train, n_classes, n_syn, 
                 edge_hidden_channels=256, nlayers=3, batch_size=1, initial='sample'):
        super(GraphSynthesizer, self).__init__()
        x_channels = x_train.shape[1]
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_syn = n_syn
        self.device = x_train.device
        self.initial = initial
        #-------------------------------------------------------------------------
        self.x_syn = nn.Parameter(torch.FloatTensor(n_syn, x_channels))
        self.y_syn, self.n_each_y, self.starts = \
            assign_each_y_proportionally(y_train, n_syn, n_classes)
        self.y_syn_batch = torch.cat([self.y_syn]*self.batch_size,dim=0)
        #-------------------------------------------------------------------------
        self.adj_mlp = nn.Sequential(
            nn.Linear(x_channels*2, edge_hidden_channels),
            nn.BatchNorm1d(edge_hidden_channels),
            nn.ReLU(),
            nn.Linear(edge_hidden_channels, edge_hidden_channels),
            nn.BatchNorm1d(edge_hidden_channels),
            nn.ReLU(),
            nn.Linear(edge_hidden_channels, 1)
        )
        #-------------------------------------------------------------------------
        feat_syn_ini = initialize_x_syn(x_train, y_train, self.y_syn, n_classes, initial)
        self.to(self.device)
        self.x_syn.data.copy_(feat_syn_ini)
        self.reset_adj_batch()
    def x_parameters(self):
        return [self.x_syn]
    def adj_parameters(self):
        return self.adj_mlp.parameters()
    def reset_adj_batch(self):
        rows = []
        cols = []
        batch = []
        for i in range(self.batch_size):
            n_neighbor = torch.zeros(self.n_syn, self.n_classes, device=self.device)
            index = torch.arange(self.n_syn, device=self.device)
            row = []
            col = []
            for row_id in range(self.n_syn):
                # for _ in range(2):
                for c in range(self.n_classes):
                    c_mask = self.y_syn==c
                    c_mask[row_id] = False
                    if c_mask.sum()==0 or n_neighbor[row_id][c]>1:
                        continue
                    link_coef = n_neighbor[c_mask,self.y_syn[row_id]]
                    selected = link_coef.argmin()
                    candidates_mask = link_coef==link_coef[selected]
                    if candidates_mask.sum()==1:
                        col_id = index[c_mask][selected].item()
                    else:
                        candidates = index[c_mask][candidates_mask]
                        col_id = candidates[torch.randperm(candidates.shape[0], device=self.device)[0]].item()
                    n_neighbor[row_id,c] += 1
                    n_neighbor[col_id,self.y_syn[row_id]] += 1
                    row.append(row_id)
                    row.append(col_id)
                    col.append(col_id)
                    col.append(row_id)
            rows.append(torch.LongTensor(row))
            cols.append(torch.LongTensor(col))
            batch.append(torch.LongTensor([i]*len(row)))
            
        self.rows = torch.cat(rows).to(self.device)
        self.cols = torch.cat(cols).to(self.device)
        self.batch = torch.cat(batch).to(self.device)
    def get_adj_t_syn(self):
        adj = torch.zeros(size=(self.batch_size, self.n_syn, self.n_syn), device=self.device)
        adj[self.batch, self.rows, self.cols] = torch.sigmoid(self.adj_mlp(
                            torch.cat([self.x_syn[self.rows], self.x_syn[self.cols]], dim=1)).flatten())
        adj = (torch.transpose(adj, 1, 2) + adj)/2
        
        adj += torch.eye(self.n_syn, device=self.device).view(1,self.n_syn,self.n_syn)
        deg = adj.sum(2)
        deg_inv = deg.pow(-1/2)
        deg_inv[torch.isinf(deg_inv)] = 0.
        adj = adj * deg_inv.view(self.batch_size, -1, 1)
        adj = adj * deg_inv.view(self.batch_size, 1, -1)
        return adj
    def reset_x(self, x_train, y_train, n_classes):
        feat_syn_ini = initialize_x_syn(x_train, y_train, self.y_syn, n_classes, self.initial)
        self.x_syn.data.copy_(feat_syn_ini)
    def __repr__(self):
        n_edge = self.rows.shape[0]/self.batch_size/2
        return f'nodes:{self.x_syn.shape}\nnumber of nodes in each class:{self.n_each_y.tolist()}\nn_edge:{n_edge}'
class FixLenList:
    def __init__(self, lenth):
        self.lenth = lenth
        self.data = []
    def append(self, element):
        self.data.append(element)
        if len(self.data) > self.lenth:
            del self.data[0]
#%% Functions in Fuxian
def match_loss(gw_syns, gw_reals):
    dis = 0
    for ig in range(len(gw_reals)):
        gw_real = gw_reals[ig]
        gw_syn = gw_syns[ig]
        dis += distance_wb(gw_real, gw_syn)

    return dis


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        pass
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def reset_bns(model, x_train, adj_t_train):
    BN_flag = False
    for module in model.modules():
        if 'BatchNorm' in module._get_name(): #BatchNorm
            BN_flag = True
    if BN_flag:
        model.train()
        model.forward(x_train, adj_t_train)
        model.eval()



