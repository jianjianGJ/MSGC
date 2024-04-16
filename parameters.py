common_arch = {'hidden_channels':256,
                'num_layers':2,
                'dropout':0.0,
                'batch_norm':False}

basic_architectures = {'SGC':common_arch,
                 'GCN':common_arch,
                 'APPNP':dict(common_arch,**{'num_linear':1})}
val_architectures = {'SGC':common_arch,
                     'SGC_rich':common_arch,
                 'GCN':common_arch,
                 'SAGE':common_arch,
                 'APPNP':dict(common_arch,**{'num_linear':1}),
                 'GAT':{'hidden_channels':128,'dropout':0.5}}
def set_args(args):
    
    args.window = 20
    args.patience = 20
    args.sizes = [10,5]
    args.batch_size_sample = 256
    args.lr_basic_model = 0.01
    args.lr_val_model = 0.01
    args.repeat = 10
    args.weight_decay = 5e-4
    
    args.basic_model_architecture = basic_architectures[args.basic_model]
    args.val_model_architecture = val_architectures[args.val_model]
    
    if args.dataset=='genius':
        #%% SGC as base
        args.standardscaler = False
        args.transductive = True
        args.edge_hidden_channels = 128
        args.epochs_initial = 600
        args.initial = 'mean'
        args.epochs_evalue = 50
        args.lr_x = 0.01
        args.lr_adj = 0.01
        args.epochs_steps = 10
        args.epochs_basic_model = 1
        if args.val_model == 'SGC':
            args.val_model = 'SGC_rich'
            args.val_model_architecture = val_architectures['SGC_rich']
    if args.dataset=='cora':
        #%% SGC as base
        args.standardscaler = False
        args.transductive = True
        args.edge_hidden_channels = 128
        args.epochs_initial = 600
        args.initial = 'mean'
        args.epochs_evalue = 50
        if args.basic_model == 'SGC':
            args.lr_x = 0.0001
            args.lr_adj = 0.0001
            args.epochs_steps = 20
            args.epochs_basic_model = 15
        #%% GCN as base
        if args.basic_model == 'GCN':
            args.lr_x = 0.001
            args.lr_adj = 0.001
            args.epochs_steps = 20
            args.epochs_basic_model = 0
        #%% SAGE as base
        if args.basic_model == 'APPNP':
            args.lr_x = 0.001
            args.lr_adj = 0.001
            args.epochs_steps = 15
            args.epochs_basic_model = 10
        if args.val_model == 'SGC':
            args.val_model = 'SGC_rich'
            args.val_model_architecture = val_architectures['SGC_rich']
        if args.val_model == 'APPNP':
            args.val_model_architecture['num_linear'] = 2
        #%%
    if args.dataset=='citeseer':
        args.standardscaler = False
        args.transductive = True
        args.edge_hidden_channels = 128
        args.epochs_initial = 600
        args.initial = 'cluster'
        args.epochs_evalue = 50
        if args.basic_model == 'SGC':
            args.lr_x = 0.0001
            args.lr_adj = 0.0001
            args.epochs_steps = 20
            args.epochs_basic_model = 15
        if args.basic_model == 'GCN':
            args.lr_x = 0.01
            args.lr_adj = 0.01
            args.epochs_steps = 15
            args.epochs_basic_model = 3
        if args.basic_model == 'APPNP':
            args.lr_x = 0.1
            args.lr_adj = 0.1
            args.epochs_steps = 10
            args.epochs_basic_model = 0
        if args.val_model == 'SGC':
            args.val_model = 'SGC_rich'
            args.val_model_architecture = val_architectures['SGC_rich']
        if args.val_model == 'APPNP':
            args.val_model_architecture['num_linear'] = 2
    if args.dataset=='arxiv':
        args.standardscaler = True
        args.transductive = True
        args.edge_hidden_channels = 256
        args.epochs_initial = 1000
        args.weight_decay = 0.
        args.val_model_architecture['dropout'] = 0.5
        args.initial = 'cluster'
        args.epochs_evalue = 300
        if args.basic_model=='SGC':
            args.basic_model = 'SGC_rich'
            args.lr_x = 0.01
            args.lr_adj = 0.01
            args.epochs_steps = 20
            args.epochs_basic_model = 3
        if args.basic_model == 'GCN':
            args.lr_x = 0.01
            args.lr_adj = 0.01
            args.epochs_steps = 10
            args.epochs_basic_model = 3
        if args.basic_model == 'APPNP':
            args.lr_x = 0.1
            args.lr_adj = 0.1
            args.epochs_steps = 10
            args.epochs_basic_model = 1
        if args.val_model == 'SGC':
            args.val_model = 'SGC_rich'
            args.val_model_architecture = val_architectures['SGC_rich']
        if args.val_model == 'APPNP':
            args.val_model_architecture['num_linear'] = 2
    if args.dataset=='flickr':
        args.standardscaler = True
        args.transductive = False
        args.edge_hidden_channels = 256
        args.epochs_initial = 600
        args.epochs_evalue = 50
        args.sizes = [15,8]
        args.initial = 'sample'
        if args.basic_model=='SGC':
            args.basic_model = 'SGC_rich'
            args.lr_x = 0.005
            args.lr_adj = 0.005
            args.epochs_steps = 10
            args.epochs_basic_model = 1
        if args.basic_model == 'GCN':
            args.lr_x = 0.005
            args.lr_adj = 0.005
            args.epochs_steps = 10
            args.epochs_basic_model = 1
        if args.basic_model == 'APPNP':
            args.lr_x = 0.1
            args.lr_adj = 0.1
            args.epochs_steps = 15
            args.epochs_basic_model = 3
        if args.val_model == 'SGC':
            args.val_model = 'SGC_rich'
            args.val_model_architecture = val_architectures['SGC_rich']
        if args.val_model == 'APPNP':
            args.val_model_architecture['num_linear'] = 2
    if args.dataset=='reddit':
        args.standardscaler = True
        args.transductive = False
        args.edge_hidden_channels = 256
        args.epochs_initial = 1000
        args.val_model_architecture['dropout'] = 0.5
        args.sizes = [15,8]
        if args.n_syn <100:
            args.initial = 'mean'
        else:
            args.initial = 'sample'
        args.epochs_evalue = 800
        if args.basic_model=='SGC':
            args.lr_x = 0.1
            args.lr_adj = 0.1
            args.epochs_steps = 10
            args.epochs_basic_model = 1
        if args.basic_model == 'GCN':
            args.lr_x = 0.1
            args.lr_adj = 0.1
            args.epochs_steps = 10
            args.epochs_basic_model = 0
        if args.basic_model == 'APPNP':
            args.lr_x = 0.1
            args.lr_adj = 0.1
            args.epochs_steps = 10
            args.epochs_basic_model = 0
            args.basic_model_architecture['num_linear'] = 2
    
    
    if args.go:
        args.num_run = 1
        args.epochs_initial = 1
        args.epochs_steps = 1
        args.val_run = 1
        args.epochs_evalue = 1
        args.repeat = 1
