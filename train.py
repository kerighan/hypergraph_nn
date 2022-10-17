from hypergraph import HyperGraph, HGNN, split_train_test
from cylouvain import best_partition
import numpy as np
import networkx as nx

# load data
G = nx.read_gexf("datasets/cora/G.gexf")
V = np.load("datasets/cora/features.npy")
labels = np.load("datasets/cora/labels.npy")
y = {n: labels[i] for i, n in enumerate(G.nodes)}
y_train, y_test = split_train_test(y, .1, random_state=None)
n_nodes = len(G.nodes)
n_labels = np.max(list(y.values())) + 1

# create hypergraph
H = HyperGraph(G, methods=["louvain", "neighbors"])

# create model
model = HGNN(n_labels=n_labels,
             n_hyperedges_type=H.n_hyperedges_type,
             embedding_dim=V.shape[1],
             hyperedge_type_dim=16,
             hyperedge_dim=256,
             node_dim=64,
             hyperedge_pooling="mean",
             node_pooling="mean",
             node_activation="tanh",
             hyperedge_activation="tanh")

# model fit
model.fit(H, y_train, V=V,
          validation_data=y_test,
          learning_rate=1e-2,
          epochs=100)
# model predict
y_pred = model.predict(H, V=V)

y_true = np.array([y[node] for node in G.nodes])
print((y_pred == y_true).mean())
print(y_true)
