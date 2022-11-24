import itertools

import networkx as nx
import numpy as np
import tensorflow as tf

algo = nx.algorithms


class HyperGraph:
    WALK_LEN = 5
    N_WALKS = 1
    LOUVAIN_RESOLUTION = 1
    ROLEWALK_METHOD = "agglomerative"
    K_CLIQUE_COMMUNITY = 4

    def __init__(self, G=None, methods=["neighbors", "louvain"]):
        self.hyperedges_type = []
        self.hyperedges = []
        self.current_hyperedges_type = 0
        if G is not None:
            self.node2id = {n: i for i, n in enumerate(G.nodes)}
            self.nodes = list(G.nodes)
            self.node2hyperedges = [[] for i in range(len(G.nodes))]
            self.n_nodes = len(G.nodes)

            if "louvain" in methods:
                self.add_louvain_hyperedges(G)
            if "infomap" in methods:
                self.add_infomap_hyperedges(G)
            if "neighbors" in methods:
                self.add_neighbors_hyperedges(G)
            if "self" in methods:
                self.add_self_hyperedges(G)
            if "rolewalk" in methods:
                self.add_rolewalk_hyperedges(G)
            if "neighbors_of_neighbors" in methods:
                self.add_neighbors_of_neighbors_hyperedges(G)
            if "random_walks" in methods:
                self.add_random_walks_hyperedges(G)
            if "onion_layers" in methods:
                self.add_onion_layers_hyperedges(G)
            if "k_clique_communities" in methods:
                self.add_k_clique_communities_hyperedges(G)
            if "isolates" in methods:
                self.add_isolates_hyperedges(G)
            if "articulation_points" in methods:
                self.add_articulation_points_hyperedges(G)
            if "min_dominating_set" in methods:
                self.add_min_dominating_set_hyperedges(G)
            if "voterank" in methods:
                self.add_voterank_hyperedges(G)
            if "max_clique" in methods:
                self.add_max_clique_hyperedges(G)
            if "vertex_cover" in methods:
                self.add_vertex_cover_hyperedges(G)
            self.n_hyperedges_type = self.current_hyperedges_type

    def add_nodes_from(self, nodes):
        self.node2id = {}
        for i, node in enumerate(nodes):
            self.node2id[node] = i
        self.node2hyperedges = [[] for i in range(len(nodes))]
        self.n_nodes = len(nodes)
        self.nodes = nodes

    def add_hyperedges_from(self, hyperedges, ids=False):
        if isinstance(hyperedges, list):
            hyperedge_index = len(self.hyperedges)
            for i, hyperedge in enumerate(hyperedges):
                if ids:
                    for node_id in hyperedge:
                        self.node2hyperedges[node_id].append(
                            hyperedge_index + i)
                    self.hyperedges.append(hyperedge)
                else:
                    h = []
                    for node in hyperedge:
                        node_id = self.node2id[node]
                        self.node2hyperedges[node_id].append(
                            hyperedge_index + i)
                        h.append(node_id)
                    self.hyperedges.append(h)
                self.hyperedges_type.append(self.current_hyperedges_type)
        elif isinstance(hyperedges, dict):
            partition2vertices = {}
            for node, cm in hyperedges.items():
                node_id = node if ids else self.node2id[node]
                partition2vertices.setdefault(cm, []).append(node_id)
            hyperedge_index = len(self.hyperedges)
            hyperedges = list(partition2vertices.values())
            for i, nodes in enumerate(hyperedges):
                for node_id in nodes:
                    self.node2hyperedges[node_id].append(i + hyperedge_index)
            self.hyperedges.extend(hyperedges)
            self.hyperedges_type.extend(
                [self.current_hyperedges_type] * len(hyperedges))

        self.current_hyperedges_type += 1
        self.n_hyperedges_type = self.current_hyperedges_type

    def add_vertex_cover_hyperedges(self, G):
        nodes = list(nx.algorithms.approximation.min_weighted_vertex_cover(G))
        self.add_hyperedges_from([nodes])

    def add_voterank_hyperedges(self, G):
        nodes = nx.voterank(G)
        self.add_hyperedges_from([nodes])

    def add_min_dominating_set_hyperedges(self, G):
        nodes = nx.algorithms.approximation.min_weighted_dominating_set(G)
        self.add_hyperedges_from([nodes])

    def add_articulation_points_hyperedges(self, G):
        nodes = list(
            nx.algorithms.components.articulation_points(G))
        self.add_hyperedges_from([nodes])

    def add_isolates_hyperedges(self, G):
        nodes = list(nx.isolates(G))
        self.add_hyperedges_from([nodes])

    def add_k_clique_communities_hyperedges(self, G):
        cm = list(algo.community.k_clique_communities(
            G, self.K_CLIQUE_COMMUNITY))
        self.add_hyperedges_from(cm)

    def add_onion_layers_hyperedges(self, G):
        try:
            partition = nx.algorithms.core.onion_layers(G)
        except nx.exception.NetworkXError:
            T = G.copy()
            T.remove_edges_from(nx.selfloop_edges(T))
            partition = nx.algorithms.core.onion_layers(T)
        self.add_hyperedges_from(partition)

    def add_max_clique_hyperedges(self, G):
        print(nx.algorithms.approximation.clique.max_clique(G))
        raise ValueError

    def add_infomap_hyperedges(self, G):
        import infomap as ip
        if G.is_directed():
            infomap_ = ip.Infomap("--two-level --directed --silent")
        else:
            infomap_ = ip.Infomap("--two-level --silent")
        network = infomap_.network
        for u, v in G.edges:
            network.addLink(self.node2id[u], self.node2id[v])
        infomap_.run()

        hyperedges = {}
        for node in infomap_.iterTree():
            if node.isLeaf():
                node_id = node.physicalId
                cm = node.moduleIndex()
                hyperedges.setdefault(cm, []).append(node_id)

        hyperedge_index = len(self.hyperedges)
        hyperedges = list(hyperedges.values())
        for i, nodes in enumerate(hyperedges):
            for node_id in nodes:
                self.node2hyperedges[node_id].append(i + hyperedge_index)

        self.hyperedges.extend(hyperedges)
        self.hyperedges_type.extend(
            [self.current_hyperedges_type] * len(hyperedges))
        self.current_hyperedges_type += 1
        self.n_hyperedges_type = self.current_hyperedges_type

    def add_random_walks_hyperedges(self, G):
        from walker import random_walks
        X = random_walks(G, n_walks=self.N_WALKS,
                         walk_len=self.WALK_LEN, p=.25, q=.25)
        hyperedge_index = len(self.hyperedges)
        for i, row in enumerate(X):
            hyperedge = []
            for node_id in row:
                self.node2hyperedges[node_id].append(i + hyperedge_index)
                hyperedge.append(node_id)
            self.hyperedges.append(hyperedge)
            self.hyperedges_type.append(self.current_hyperedges_type)
        self.current_hyperedges_type += 1
        self.n_hyperedges_type = self.current_hyperedges_type

    def add_rolewalk_hyperedges(self, G):
        from rolewalk import RoleWalk
        y = RoleWalk(walk_len=3).fit_predict(G, method=self.ROLEWALK_METHOD)
        hyperedges = {}
        for node_id, cm in enumerate(y):
            hyperedges.setdefault(cm, []).append(node_id)

        hyperedge_index = len(self.hyperedges)
        hyperedges = list(hyperedges.values())
        for i, nodes in enumerate(hyperedges):
            for node_id in nodes:
                self.node2hyperedges[node_id].append(i + hyperedge_index)

        self.hyperedges.extend(hyperedges)
        self.hyperedges_type.extend(
            [self.current_hyperedges_type] * len(hyperedges))
        self.current_hyperedges_type += 1
        self.n_hyperedges_type = self.current_hyperedges_type

    def add_louvain_hyperedges(self, G):
        try:
            from cylouvain import best_partition
        except ImportError:
            from community import best_partition
        partition = best_partition(G, resolution=self.LOUVAIN_RESOLUTION)
        self.add_hyperedges_from(partition)

    def add_neighbors_hyperedges(self, G):
        hyperedge_index = len(self.hyperedges)
        for i, node in enumerate(G.nodes):
            node_id = self.node2id[node]
            hyperedge = []
            # include node itself in hyperedge
            hyperedge.append(node_id)
            self.node2hyperedges[node_id].append(i + hyperedge_index)

            for neighbor in G.neighbors(node):
                nb_id = self.node2id[neighbor]
                if nb_id == node_id:
                    continue
                hyperedge.append(nb_id)
                self.node2hyperedges[nb_id].append(i + hyperedge_index)

            self.hyperedges.append(hyperedge)
            self.hyperedges_type.append(self.current_hyperedges_type)
        self.current_hyperedges_type += 1
        self.n_hyperedges_type = self.current_hyperedges_type

    def add_neighbors_of_neighbors_hyperedges(self, G):
        hyperedge_index = len(self.hyperedges)
        for i, node in enumerate(G.nodes):
            node_id = self.node2id[node]
            hyperedge = set()
            # include node itself in hyperedge
            hyperedge.add(node_id)
            self.node2hyperedges[node_id].append(i + hyperedge_index)

            neighbors = set(G.neighbors(node))
            for neighbor in neighbors:
                for neighbor_2 in G.neighbors(neighbor):
                    if neighbor_2 in neighbors:
                        continue
                    nb_id = self.node2id[neighbor_2]
                    hyperedge.add(nb_id)
                    self.node2hyperedges[nb_id].append(i + hyperedge_index)

            self.hyperedges.append(list(hyperedge))
            self.hyperedges_type.append(self.current_hyperedges_type)
        self.current_hyperedges_type += 1
        self.n_hyperedges_type = self.current_hyperedges_type

    def add_self_hyperedges(self, G):
        hyperedge_index = len(self.hyperedges)
        for i, node in enumerate(G.nodes):
            node_id = self.node2id[node]
            self.node2hyperedges[node_id].append(i + hyperedge_index)
            self.hyperedges.append([node_id])
            self.hyperedges_type.append(self.current_hyperedges_type)
        self.current_hyperedges_type += 1
        self.n_hyperedges_type = self.current_hyperedges_type

    @property
    def V2E(self):
        return tf.ragged.constant(
            [self.node2hyperedges[i] for i in range(self.n_nodes)],
            dtype=np.int32)

    @property
    def E2V(self):
        return tf.ragged.constant(self.hyperedges, dtype=np.int32)

    @property
    def n_hyperedges(self):
        return len(self.hyperedges)

    def __repr__(self):
        return (f"Hypergraph with {self.n_nodes} nodes and "
                f"{self.n_hyperedges} hyperedges")

    def save(self, filename):
        import pickle
        pickle.dump(self, open(filename, "wb"))

    @staticmethod
    def load(filename):
        import pickle
        return pickle.load(open(filename, "rb"))

    def get_hyperedges_from_nodes(self, nodes):
        edges = set()
        for node in nodes:
            for hyperedge in self.node2hyperedges[node]:
                edges.add(hyperedge)
        return edges

    def get_nodes_from_hyperedges(self, edges):
        nodes = set()
        for edge in edges:
            for node in self.hyperedges[edge]:
                nodes.add(node)
        return nodes

    def get_batch(self, batch_nodes, n_layers):
        batch_nodes_id = {self.node2id[n] for n in batch_nodes}
        edges = self.get_hyperedges_from_nodes(batch_nodes_id)
        nodes = batch_nodes_id.union(self.get_nodes_from_hyperedges(edges))
        if n_layers >= 2:
            edges = self.get_hyperedges_from_nodes(nodes)
            nodes = nodes.union(self.get_nodes_from_hyperedges(edges))

        edge2id = {e: i for i, e in enumerate(edges)}
        node2id = {n: i for i, n in enumerate(nodes)}
        id2node = {i: n for n, i in node2id.items()}
        id2edge = {i: e for e, i in edge2id.items()}
        nodes_slice = [node2id[self.node2id[n]] for n in batch_nodes]
        nodes_list = [id2node[i] for i in range(len(nodes))]

        V2E_batch = tf.ragged.constant(
            [[edge2id[e] for e in self.node2hyperedges[id2node[i]]
              if e in edge2id]
             for i in range(len(nodes))], dtype=np.int32)
        E2V_batch = tf.ragged.constant(
            [[node2id[n] for n in self.hyperedges[id2edge[i]] if n in node2id]
             for i in range(len(edges))],
            dtype=np.int32)
        hyperedges_type = np.array(
            [self.hyperedges_type[id2edge[i]]
             for i in range(len(edges))],
            dtype=np.int32)
        return (
            batch_nodes, nodes_slice, nodes_list,
            V2E_batch, E2V_batch, hyperedges_type)
