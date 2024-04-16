#%%
from typing import Tuple
from torch_sparse import fill_diag, sum as sparsesum, mul
from sklearn.preprocessing import StandardScaler
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Flickr, Reddit2, LINKXDataset
from ogb.nodeproppred import PygNodePropPredDataset
#%%
def index2mask(idx, size):
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask
def gen_masks(y, train_per_class: int = 20, val_per_class: int = 30,
              num_splits: int = 20):
    num_classes = int(y.max()) + 1

    train_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        perm = torch.stack(
            [torch.randperm(idx.size(0)) for _ in range(num_splits)], dim=1)
        idx = idx[perm]

        train_idx = idx[:train_per_class]
        train_mask.scatter_(0, train_idx, True)
        val_idx = idx[train_per_class:train_per_class + val_per_class]
        val_mask.scatter_(0, val_idx, True)

    test_mask = ~(train_mask | val_mask)

    return train_mask, val_mask, test_mask
def gcn_norm(adj_t, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1., dtype=dtype)
    if add_self_loops:
        adj_t = fill_diag(adj_t, fill_value)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t
#%%

def get_genius(root: str) -> Tuple[Data, int, int]:
    # 2 classes, 421961 nodes, splits 168784/126588/126589
    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    dataset = LINKXDataset(f'{root}/LINKXDataset', 'genius', transform=transform)
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    index = torch.randperm(data.x.shape[0])
    data.train_mask = torch.zeros(index.shape[0], dtype=torch.bool)
    data.val_mask = torch.zeros(index.shape[0], dtype=torch.bool)
    data.test_mask = torch.zeros(index.shape[0], dtype=torch.bool)
    data.train_mask[index[:int(index.shape[0]*0.4)]] = True
    data.val_mask[index[int(index.shape[0]*0.4):int(index.shape[0]*0.7)]] = True
    data.test_mask[index[int(index.shape[0]*0.7):]] = True
    return data, dataset.num_features, dataset.num_classes

def get_planetoid(root: str, name: str) -> Tuple[Data, int, int]:
    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    dataset = Planetoid(f'{root}/Planetoid', name, transform=transform)
    return dataset[0], dataset.num_features, dataset.num_classes



def get_arxiv(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-arxiv', f'{root}/OGB',
                                     pre_transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.node_year = None
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes


def get_flickr(root: str) -> Tuple[Data, int, int]:
    dataset = Flickr(f'{root}/Flickr', pre_transform=T.ToSparseTensor())
    return dataset[0], dataset.num_features, dataset.num_classes


def get_reddit(root: str) -> Tuple[Data, int, int]:
    dataset = Reddit2(f'{root}/Reddit2', pre_transform=T.ToSparseTensor())
    data = dataset[0]
    return data, dataset.num_features, dataset.num_classes





def get_data(root: str, name: str) -> Tuple[Data, int, int]:
    if name.lower() in ['cora', 'citeseer']:
        return get_planetoid(root, name)
    elif name.lower() == 'reddit':
        return get_reddit(root)
    elif name.lower() == 'flickr':
        return get_flickr(root)
    elif name.lower() in ['ogbn-arxiv', 'arxiv']:
        return get_arxiv(root)
    elif name.lower() == 'genius':
        return get_genius(root)
    else:
        raise NotImplementedError
#'./GCond_data/'
def load_data(data_name, root='/home/gj/SpyderWorkSpace/data/', transductive=True,
              add_selfloop=True, norm=True, standardscaler=False):
    
    data, num_features, num_classes = get_data(root,data_name)
    
    if transductive:
        if add_selfloop:
            data.adj_t.set_value_(torch.ones(data.adj_t.nnz()),layout='coo')
            data.adj_t = data.adj_t.set_diag(torch.ones(data.adj_t.size(0))*1)
        if norm:
            data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        if standardscaler:
            scaler = StandardScaler()
            scaler.fit(data.x)
            data.x = torch.FloatTensor(scaler.transform(data.x))
        data.x_train = data.x_val = data.x_test = data.x
        data.adj_t_train = data.adj_t_val = data.adj_t_test = data.adj_t
        data.y_train = data.y.clone()
        data.y_val = data.y.clone()
        data.y_test = data.y.clone()
        data.y_train[~data.train_mask] = -1
        data.y_val[~data.val_mask] = -1
        data.y_test[~data.test_mask] = -1
        
    else: #inductive
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        
        data.adj_t_train = data.adj_t[train_mask,train_mask]
        data.adj_t_val = data.adj_t[val_mask,val_mask]
        data.adj_t_test = data.adj_t[test_mask,test_mask]
        
        if add_selfloop:
            data.adj_t_train.set_value_(torch.ones(data.adj_t_train.nnz()),layout='coo')
            data.adj_t_train = data.adj_t_train.set_diag(torch.ones(data.adj_t_train.size(0))*1)
            data.adj_t_val.set_value_(torch.ones(data.adj_t_val.nnz()),layout='coo')
            data.adj_t_val = data.adj_t_val.set_diag(torch.ones(data.adj_t_val.size(0))*1)
            data.adj_t_test.set_value_(torch.ones(data.adj_t_test.nnz()),layout='coo')
            data.adj_t_test = data.adj_t_test.set_diag(torch.ones(data.adj_t_test.size(0))*1)
        if norm:
            data.adj_t_train = gcn_norm(data.adj_t_train, add_self_loops=False)
            data.adj_t_val = gcn_norm(data.adj_t_val, add_self_loops=False)
            data.adj_t_test = gcn_norm(data.adj_t_test, add_self_loops=False)
        
        ##########################################################################
        if standardscaler:
            scaler = StandardScaler()
            scaler.fit(data.x[train_mask])
            data.x = torch.FloatTensor(scaler.transform(data.x))
        ##########################################################################
        data.x_train = data.x[train_mask]
        data.x_val = data.x[val_mask]
        data.x_test = data.x[test_mask]
        
        
        
        
        data.y_train = data.y[train_mask]
        data.y_val = data.y[val_mask]
        data.y_test = data.y[test_mask]
        
    del data.train_mask, data.val_mask, data.test_mask, data.x, data.y, data.adj_t
    return data, num_features, num_classes

















