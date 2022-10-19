import networkx as nx
import numpy as np
from rolewalk import RoleWalk

from hypergraph import (HyperGNN, HyperGraph, get_hyperedges_from_label_matrix,
                        split_train_test)

# load data
dataset = "cora"
G = nx.read_gexf(f"datasets/{dataset}/G.gexf")
V = np.load(f"datasets/{dataset}/features.npy")
labels = np.load(f"datasets/{dataset}/labels.npy")
n_labels = np.max(labels) + 1
# create training/testing set
y = {n: labels[i] for i, n in enumerate(G.nodes)}
y_train, y_test = split_train_test(y, .4, random_state=1)
print(len(y_train), len(y_test), n_labels)

# create hypergraph from graph
H = HyperGraph(G, methods=["neighbors", "louvain"])
# add hyperedges from the feature matrix
# H.add_hyperedges_from(get_hyperedges_from_label_matrix(V))
# add structural role embedding
X = RoleWalk(embedding_dim=2).fit_transform(G)
V = np.hstack([X, V])

# create model
model = HyperGNN(n_labels=n_labels,
                 n_hyperedges_type=H.n_hyperedges_type,
                 embedding_dim=V.shape[1],
                 hyperedge_type_dim=8,
                 hyperedge_dim=768,
                 node_dim=768,
                 node_activation="tanh",
                 hyperedge_activation="tanh")
# model fit
model.fit(H, y_train, V=V,
          validation_data=y_test,
          learning_rate=1e-3,
          epochs=40)

# persist model to disk
model.save("model.p")
# model load
model = HyperGNN.load("model.p")

# predict on a set of potentially different hypergraph and features
y_pred = model.predict(H, V=V)
print(y_pred)
