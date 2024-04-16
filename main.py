import os
import argparse
from tqdm import tqdm
import torch
from copy import deepcopy
from utils_data import load_data
import torch.nn.functional as F
from graphsyn import (GraphSynthesizer, NeighborSamplerForClass, 
                      match_loss, reset_bns,
                      FixLenList)
from utils import get_GNN, train_model, seed, evalue_model
from parameters import set_args
#%% Parameters

argparser = argparse.ArgumentParser("Test",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--dataset", type=str, default='cora')
argparser.add_argument("--n-syn", type=int, default=35)
argparser.add_argument("--batch-size-syn", type=int, default=16)
argparser.add_argument("--basic-model", type=str, default='SGC')
argparser.add_argument("--val-model", type=str, default='GCN')
argparser.add_argument("--num-run", type=int, help="running times", default=1)
argparser.add_argument("--val-run", type=int, help="running times", default=1)
argparser.add_argument("--only_evalue", action='store_true', default=False)
argparser.add_argument("--go", action='store_true', default=False)
argparser.add_argument("--hyid", type=int, default=1)
args = argparser.parse_args()
# args.go = True
set_args(args)
if not os.path.exists('./data_syn/'):
    os.makedirs('./data_syn/')
if args.transductive:
    tem = 'Tra'
else:
    tem = 'Ind'
version = f'{args.dataset}-{tem}-{args.basic_model}-{args.n_syn}-{args.batch_size_syn}'
print(version)
save_path = f'./data_syn/{version}'
if not os.path.exists(save_path) and not args.go:
    os.makedirs(save_path)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
#%% Data
data, n_node_feats, n_classes = load_data(args.dataset, transductive=args.transductive, standardscaler=args.standardscaler)
adj_t_train, x_train, y_train = data.adj_t_train, data.x_train, data.y_train
adj_t_train, x_train, y_train = adj_t_train.to(device), x_train.to(device), y_train.to(device)
adj_t_val, x_val, y_val = data.adj_t_val, data.x_val, data.y_val 
adj_t_val, x_val, y_val = adj_t_val.to(device), x_val.to(device), y_val.to(device)
adj_t_test, x_test, y_test = data.adj_t_test, data.x_test, data.y_test
adj_t_test, x_test, y_test = adj_t_test.to(device), x_test.to(device), y_test.to(device)
#%% Train
    #######################实例化图生成器及其优化器##################################
acc_tests = []
for run_id in range(args.num_run):
    file_path = f'{save_path}/run_{run_id+1}/'
    if os.path.exists(f'{file_path}/adj_t_syn.tensor') and args.only_evalue:
        best_adj_t_syn = torch.load(f'{file_path}/adj_t_syn.tensor').to(device)
        y_syn = torch.load(f'{file_path}/y_syn.tensor').to(device)
        best_x_syn = torch.load(f'{file_path}/x_syn.tensor').to(device)
    else:
        seed(run_id)
        neighor_sampler = NeighborSamplerForClass(adj_t_train, y_train, n_classes, sizes=args.sizes)
        graphsyner = GraphSynthesizer(x_train, y_train, adj_t_train, n_classes, edge_hidden_channels=args.edge_hidden_channels,
                                      n_syn=args.n_syn, batch_size=args.batch_size_syn, initial=args.initial)
        print(graphsyner)
        y_syn = graphsyner.y_syn_batch
        n_each_y = graphsyner.n_each_y
        basic_model = get_GNN(args.basic_model)(in_channels=n_node_feats,
                                            out_channels=n_classes,
                                            **args.basic_model_architecture).to(device)
        
        graphsyner.reset_adj_batch()
        graphsyner.reset_x(x_train, y_train, n_classes)
        optimizer_x = torch.optim.Adam(graphsyner.x_parameters(), lr=args.lr_x)
        optimizer_adj = torch.optim.Adam(graphsyner.adj_parameters(), lr=args.lr_adj)
        ##############################在不同的初始化下进行优化###############################
        smallest_loss = 99999.
        losses = FixLenList(args.window)
        x_syns = FixLenList(args.window)
        adj_t_syns = FixLenList(args.window)
        for initial_i in tqdm(range(args.epochs_initial),desc=f'running {run_id}', ncols=80):
            basic_model.reset_parameters()
            optimizer_basic_model = torch.optim.Adam(basic_model.parameters(), lr=args.lr_basic_model)
            loss_avg = 0
            ##########################在该初始化下，交替进行图优化与模型优化#######################
            for step_syn in range(args.epochs_steps):
                reset_bns(basic_model, x_train, adj_t_train)
                basic_model.eval()# fix basic_model while optimizing graphsyner
                ######################先进行图优化#####################################
                adj_t_syn = graphsyner.get_adj_t_syn()
                x_syn = graphsyner.x_syn
                loss = 0.
                for c in range(n_classes):
                    #------------------------------------------------------------------
                    batch_size, n_id, adjs = neighor_sampler.sample_from_class(
                            c, batch_size=args.batch_size_sample)
                    output = basic_model(x_train[n_id], adjs)
                    loss_real = F.nll_loss(output, y_train[n_id[:batch_size]])
                    gw_reals = torch.autograd.grad(loss_real, basic_model.parameters())
                    gw_reals = list((_.detach().clone() for _ in gw_reals))
                    #------------------------------------------------------------------
                    output_syn = basic_model(x_syn, adj_t_syn)
                    loss_syn = F.nll_loss(output_syn[y_syn==c], y_syn[y_syn==c])
                    gw_syns = torch.autograd.grad(loss_syn, basic_model.parameters(), create_graph=True)
                    #------------------------------------------------------------------
                    coeff = n_each_y[c] / args.n_syn
                    ml = match_loss(gw_syns, gw_reals)
                    loss += coeff  * ml
                loss_avg += loss.item()
                optimizer_x.zero_grad()
                optimizer_adj.zero_grad()
                loss.backward()
                if initial_i % 50 < 10:
                    optimizer_adj.step()
                else:
                    optimizer_x.step()
                
                x_syn = graphsyner.x_syn.detach()
                adj_t_syn = graphsyner.get_adj_t_syn().detach()
                #########################再进行模型参数优化###########################
                train_model(basic_model, x_syn, adj_t_syn, y_syn, optimizer_basic_model, 
                            args.epochs_basic_model, loss_fn=F.nll_loss)
            #################完成了某个初始化下的优化，下面进行记录，评价###########################
            loss_avg /= (n_classes*args.epochs_steps)
            losses.append(loss_avg)
            x_syns.append(x_syn.clone())
            adj_t_syns.append(adj_t_syn.clone())
            loss_window = sum(losses.data)/len(losses.data)
            if loss_window < smallest_loss:
                patience = 0
                smallest_loss = loss_window
                best_x_syn = sum(x_syns.data)/len(x_syns.data)
                best_adj_t_syn = sum(adj_t_syns.data)/len(adj_t_syns.data)
                print(f'{initial_i} loss:{smallest_loss:.4f} feat:{x_syn.sum().item():.4f} adj:{adj_t_syn.sum().item():.4f}')
            else:
                patience += 1
                if patience >= args.patience:
                    break
        
        if not os.path.exists(file_path) and not args.go:
            os.makedirs(file_path)
        if not args.go:
            torch.save(best_adj_t_syn, f'{file_path}/adj_t_syn.tensor')
            torch.save(best_x_syn, f'{file_path}/x_syn.tensor')    
            torch.save(y_syn, f'{file_path}/y_syn.tensor')  
#%% Evaluation
    
    val_mask = y_val>=0
    for val_id in range(args.val_run):
        best_acc = 0.
        val_model = get_GNN(args.val_model)(in_channels=n_node_feats,
                                            out_channels=n_classes,
                                            **args.val_model_architecture).to(device)
        for i in tqdm(range(args.epochs_evalue*args.repeat),desc=f'evalation {val_id}', ncols=80):
            if i%args.epochs_evalue == 0:
                  optimizer_val_model = torch.optim.Adam(val_model.parameters(), lr=args.lr_val_model,weight_decay=args.weight_decay)
                  val_model.reset_parameters()
            val_model.train()
            optimizer_val_model.zero_grad()
            logits_train = val_model(best_x_syn, best_adj_t_syn)
            loss_train = F.nll_loss(logits_train, y_syn)
            loss_train.backward()
            optimizer_val_model.step()
            with torch.no_grad():
                val_model.eval()
                logits_val = val_model(x_val, adj_t_val)[val_mask]
                acc = (logits_val.argmax(1) == y_val[val_mask]).sum()/logits_val.shape[0]
                if acc > best_acc:
                    best_acc = acc
                    # print(f'train loss:{loss_train.item():.4}, acc:{acc:.4}')
                    weights = deepcopy(val_model.state_dict())
        # if not args.go:
        #     torch.save(weights, f'{file_path}/{args.val_model}_{val_id}.model')
        val_model.load_state_dict(weights)
        loss_test, acc_test = evalue_model(val_model, x_test, adj_t_test, y_test, 
                                                loss_fn=F.nll_loss, critic='acc')
        acc_tests.append(acc_test)
acc_tests = torch.tensor(acc_tests)
std, mean = torch.std_mean(acc_tests)
std, mean = std.item(), mean.item()
log_path = './data_syn/result.txt'
version += f'-{args.val_model}'
with open(log_path,'a') as f:
    if not args.go:
        log = f'{version}:{mean:.4f} {std:.4f}\n'
    else:
        log = f'**{version}:{mean:.4f} {std:.4f}\n'
    f.write(log)
print(acc_tests)












