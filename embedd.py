import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import umap
from rolewalk import RoleWalk

from data import load_citeseer, load_cora, load_pubmed
from hypergnn import HyperGNN, HyperGraph, split_train_test

# load data
G, V, y = load_cora()
y_train, y_test = split_train_test(y, .4, random_state=0)

# create hypergraph from graph
H = HyperGraph(G,
               methods=[
                   "neighbors",
                   "louvain"
               ])

# create and fit model
model = HyperGNN(hyperedge_type_dim=16,
                 hyperedge_dim=256,
                 node_dim=256,
                 node_activation="tanh",
                 hyperedge_activation="tanh",
                 n_layers=1)
X = model.embedd(H, V,
                 learning_rate=1e-3,
                 optimizer="nadam",
                 epochs=80)

X = umap.UMAP().fit_transform(X)
plt.scatter(X[:, 0], X[:, 1], c=[y[node] for node in G.nodes], s=1.5)
plt.show()
