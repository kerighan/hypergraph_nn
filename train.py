import networkx as nx
import numpy as np
from rolewalk import RoleWalk

from hypergnn import (HyperGNN, HyperGraph, get_hyperedges_from_label_matrix,
                      split_train_test)

# load data
dataset = "cora"
G = nx.read_gexf(f"datasets/{dataset}/G.gexf")
V = np.load(f"datasets/{dataset}/features.npy")
labels = np.load(f"datasets/{dataset}/labels.npy")
n_labels = np.max(labels) + 1
# create training/testing dictionaries
y = {n: labels[i] for i, n in enumerate(G.nodes)}
y_train, y_test = split_train_test(y, .4, random_state=1)

# create hypergraph from graph
H = HyperGraph(G, methods=["neighbors", "louvain"])

# create and fit model
model = HyperGNN(hyperedge_type_dim=32,
                 hyperedge_dim=512,
                 node_dim=512,
                 node_activation="leaky_relu",
                 hyperedge_activation="leaky_relu",
                 attention_activation="leaky_relu",
                 n_layers=2)
model.fit(H, V, y_train,
          validation_data=y_test,
          learning_rate=1e-3,
          optimizer="nadam",
          epochs=40)

model.save("model.p")  # persist model to disk
model = HyperGNN.load("model.p")  # model load

# predict on a set of potentially different hypergraph and features
y_pred = model.predict(H, V)
