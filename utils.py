import os
import torch
import random
import numpy as np
import math
from models import GCN, SGC, SGC_rich, SAGE, APPNP, GAT
import torch.nn.functional as F
from copy import deepcopy
#%%

def seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_GNN(modelname):
    if modelname=='GCN':
        GNN = GCN
    elif modelname=='GAT':
        GNN = GAT
    elif modelname=='APPNP':
        GNN = APPNP
    elif modelname=='SAGE':
        GNN = SAGE
    elif modelname=='SGC':
        GNN = SGC
    elif modelname=='SGC_rich':
        GNN = SGC_rich
    return GNN

def evalue_model(model, x_test, adj_t_test, y_test, loss_fn=F.nll_loss, critic='acc'):
    model.eval()
    test_mask = y_test>=0
    logits = model(x_test, adj_t_test)[test_mask]
    loss = loss_fn(logits, y_test[test_mask])
    if critic=='acc':
        result = (logits.argmax(1) == y_test[test_mask]).sum()/logits.shape[0]
    else:
        raise Exception(f'Unimplemented critic {critic}!!')
    return loss.item(), result
def train_model(model, x_syn, adj_t_syn, y_syn, optimizer, epochs, loss_fn=F.nll_loss):
    losses = []
    for i in range(epochs):
        optimizer.zero_grad()
        logits = model(x_syn, adj_t_syn)
        loss = loss_fn(logits, y_syn)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
def train_model_ealystop(model, x_syn, adj_t_syn, y_syn, x_val, adj_t_val, y_val, 
                         optimizer, epochs, loss_fn=F.nll_loss):
    val_mask = y_val >= 0
    best_acc = 0.
    loss_train_list = []
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_syn, adj_t_syn)
        loss = loss_fn(logits, y_syn)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            logits_val = model(x_val, adj_t_val)[val_mask]
            acc = (logits_val.argmax(1) == y_val[val_mask]).sum()/logits_val.shape[0]
            if acc > best_acc:
                loss_train_list.append(loss.item())
                best_acc = acc
                # print(f'train loss:{loss_train.item():.4}, acc:{acc:.4}')
                weights = deepcopy(model.state_dict())
    model.load_state_dict(weights)
    return loss_train_list