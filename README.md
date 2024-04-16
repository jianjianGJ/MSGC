# MSGC
Code for KBS paper "Multiple Sparse Graphs Condensation".

**Abstract**—The growing graph scale presents challenges for applying graph neural networks (GNNs) practically. Recently, Graph
condensation (GCond) was proposed to condense the original large-scale graph into a small-scale graph. The goal is to
make GNNs trained on the condensed graph achieve the performance close to those trained on the original graph. GCond
achieves satisfactory performance on some datasets. However, GCond uses a single fully connected graph to model the
condensed graph, which limits the diversity of embeddings obtained, especially when there are few synthetic nodes. We
propose Multiple Sparse Graphs Condensation (MSGC), which condenses the original large-scale graph into multiple
small-scale sparse graphs. MSGC takes standard neighborhood patterns as the basic substructures and can construct
various connection schemes. Correspondingly, GNNs can obtain multiple sets of embeddings, which significantly enriches
the diversity of embeddings. Experiments show that compared with GCond and other baselines, MSGC has significant
advantages under the same condensed graph scale. MSGC can retain nearly 100% performance on Flickr and Citeseer
datasets while reducing their graph scale by over 99.0%.

## Environment configuration
* python          3.8
* torch           1.12.1+cu113
* torch_geometric 2.1.0
* torch_sparse    0.6.15
* torch_scatter   2.0.9
* prettytable     3.4.1
* sklearn         1.1.2
* tqdm            4.64.0
* numpy           1.21.5

## Dataset
In `utils_data.py`, change argument `root` in function `load_data` to your path.

The dataset will be downloaded automatically. If the download fails, you can view the source code of `torch_geometric.datasets` and update the url.

## Run
`python -u main.py --dataset=cora --basic-model=SGC --val-model=GCN --n-syn=35 --batch-size-syn=16--num-run=5 --val-run=5 `

* basic-model: The relay model in gradient matching loss calculation.
* val-model: The model to evaluate the condensed graph.
* n-syn: The number of synthetic nodes in the condensed graph.
* batch-size-syn: The number of synthetic adjacency matrices in the condensed graph.
* num-run: To run the condensation process multiple times.
* val-run: To evalue the condensed graph multiple times.

Modify `parameters.py` for more detailed setup.
