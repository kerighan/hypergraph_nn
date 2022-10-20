import networkx as nx
import numpy as np
import tensorflow as tf
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
H = HyperGraph(G, methods=["neighbors", "louvain", "infomap"])

#
preprocessor = tf.keras.Sequential()
preprocessor.add(tf.keras.layers.Dense(512, activation="tanh"))

# create model
model = HyperGNN(hyperedge_type_dim=16,
                 hyperedge_dim=768,
                 node_dim=768,
                 node_activation="tanh",
                 hyperedge_activation="tanh")
# model fit
model.fit(H, V, y_train,
          preprocessor=preprocessor,
          validation_data=y_test,
          learning_rate=1e-3,
          epochs=40)

# persist model to disk
model.save("model.p")
# model load
model = HyperGNN.load("model.p")

# predict on a set of potentially different hypergraph and features
y_pred = model.predict(H, V=V)
