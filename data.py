import networkx as nx
import numpy as np
from stellargraph import datasets


def load_deezer():
    import json
    data = json.load(open("datasets/deezer/deezer_edges.json"))


def load_pubmed():
    dataset = datasets.PubMedDiabetes()
    graph, _subjects = dataset.load()

    G = graph.to_networkx()
    G = nx.Graph(G)
    V = np.array(graph.node_features())
    y = _subjects.to_dict()
    return G, V, y


def load_cora():
    dataset = datasets.Cora()
    graph, _subjects = dataset.load()

    G = graph.to_networkx()
    G = nx.Graph(G)
    V = np.array(graph.node_features())
    y = _subjects.to_dict()
    y = {}
    label2id = {}
    for n, label in _subjects.items():
        if label in label2id:
            y[n] = label2id[label]
        else:
            i = len(label2id)
            label2id[label] = i
            y[n] = i
    return G, V, y


def load_citeseer():
    dataset = datasets.CiteSeer()
    graph, _subjects = dataset.load()

    G = graph.to_networkx()
    G = nx.Graph(G)
    V = np.array(graph.node_features())
    y = {}
    label2id = {}
    for n, label in _subjects.to_dict().items():
        if label in label2id:
            y[n] = label2id[label]
        else:
            i = len(label2id)
            label2id[label] = i
            y[n] = i
    return G, V, y


def load_blogcatalog3():
    dataset = datasets.AIFB()
    graph, _subjects = dataset.load()
    print(_subjects)

    G = graph.to_networkx()
    G = nx.Graph(G)
    V = np.array(graph.node_features())
    y = {}
    label2id = {}
    for n, label in _subjects.to_dict().items():
        if label in label2id:
            y[n] = label2id[label]
        else:
            i = len(label2id)
            label2id[label] = i
            y[n] = i
    return G, V, y
