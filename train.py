import networkx as nx
import numpy as np
from rolewalk import RoleWalk

from data import load_citeseer, load_cora, load_pubmed
from hypergnn import HyperGNN, HyperGraph, split_train_test

# load data
G, V, y = load_pubmed()
y_train, y_test = split_train_test(y, .4, random_state=1)

# create hypergraph from graph
H = HyperGraph(G, methods=[
    "neighbors",
    # "rolewalk",
    "louvain",
    "onion_layers"
    # "infomap"
])


# create and fit model
model = HyperGNN(hyperedge_type_dim=32,
                 hyperedge_dim=512,
                 node_dim=512,
                 attention_dim=256,
                 node_activation="tanh",
                 hyperedge_activation="tanh",
                 pooling="key_query",
                 n_layers=2)
model.fit(H, V, y_train,
          validation_data=y_test,
          learning_rate=1e-3,
          optimizer="nadam",
          batch_size=None,
          epochs=30)

model.save("model.p")  # persist model to disk
model = HyperGNN.load("model.p")  # model load

# predict on a set of potentially different hypergraph and features
y_pred = model.predict(H, V)
