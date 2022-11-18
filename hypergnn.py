import networkx as nx
import numpy as np
import tensorflow as tf
from numpy.random import normal
from tensorflow.keras import layers
from tqdm import tqdm

algo = nx.algorithms


class HyperGraph:
    WALK_LEN = 10
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
        from cylouvain import best_partition
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

    def save(self, filename):
        import pickle
        pickle.dump(self, open(filename, "wb"))

    @staticmethod
    def load(filename):
        import pickle
        return pickle.load(open(filename, "rb"))


class HyperGNN:
    def __init__(
        self,
        hyperedge_type_dim=4,
        hyperedge_dim=32,
        node_dim=32,
        hyperedge_activation="relu",
        node_activation="relu",
        attention_activation="relu",
        n_layers=1
    ):
        # dimensionality
        self.hyperedge_type_dim = hyperedge_type_dim
        self.hyperedge_dim = hyperedge_dim
        self.node_dim = node_dim
        self.trainable_variables = []
        self.n_layers = n_layers

        # activations
        self.hyperedge_activation = (
            tf.keras.activations.get(hyperedge_activation))
        self.node_activation = tf.keras.activations.get(node_activation)
        self.attention_activation = (
            tf.keras.activations.get(attention_activation))

    def build(self):
        # dimensionalities
        E_W_in = self.hyperedge_type_dim + self.embedding_dim
        V_W_in = self.embedding_dim + self.hyperedge_dim

        # hyperedges embedding
        self.E_type = glorot_normal(shape=(self.n_hyperedges_type,
                                           self.hyperedge_type_dim))
        self.E_W = glorot_normal(shape=(E_W_in, self.hyperedge_dim))
        self.E_b = glorot_normal(shape=(1, self.hyperedge_dim))
        self.E_att = glorot_normal(shape=(self.embedding_dim, 1))
        self.E_att_bias = tf.Variable(np.zeros(1, dtype=np.float32))
        self.E_temperature = tf.Variable(np.ones(1, dtype=np.float32))
        self.trainable_variables.extend([self.E_att, self.E_temperature])

        # nodes embedding
        self.V_W = glorot_normal(shape=(V_W_in, self.node_dim))
        self.V_b = glorot_normal(shape=(1, self.node_dim))
        self.V_att = glorot_normal(shape=(self.hyperedge_dim, 1))
        self.V_att_bias = tf.Variable(np.zeros(1, dtype=np.float32))
        self.V_temperature = tf.Variable(np.ones(1, dtype=np.float32))

        # classification weights
        self.W = glorot_normal(shape=(self.node_dim, self.n_labels))

        # list training variables
        # self.trainable_variables = [
        #     self.E_type, self.E_W, self.E_b,
        #     self.E_att, self.E_att_bias, self.E_temperature,
        #     self.V_W, self.V_b,
        #     self.V_att, self.V_att_bias, self.V_temperature,
        #     self.W]
        self.trainable_variables = [
            self.E_type, self.E_W, self.E_b,
            self.V_W, self.V_b,
            self.W]
        self.attention_E_1 = get_multihead_self_attention(self.embedding_dim)
        raise ValueError
        self.attention_V_1 = get_weighted_attention()
        self.trainable_variables.extend(self.attention_E_1.trainable_variables)
        self.trainable_variables.extend(self.attention_V_1.trainable_variables)

        if self.n_layers >= 2:
            # dimensionalities
            E_2_W_in = self.hyperedge_type_dim + self.node_dim
            V_2_W_in = self.node_dim + self.hyperedge_dim

            # 2nd layer hyperedges embedding
            self.E_2_att = glorot_normal(shape=(self.node_dim, 1))
            self.E_2_att_bias = tf.Variable(np.zeros(1, dtype=np.float32))
            self.E_2_temperature = tf.Variable(np.ones(1, dtype=np.float32))
            self.E_2_W = glorot_normal(shape=(E_2_W_in, self.hyperedge_dim))
            self.E_2_b = glorot_normal(shape=(1, self.hyperedge_dim))

            # 2nd layer nodes embedding
            self.V_2_W = glorot_normal(shape=(V_2_W_in, self.node_dim))
            self.V_2_b = glorot_normal(shape=(1, self.node_dim))
            self.V_2_att = glorot_normal(shape=(self.hyperedge_dim, 1))
            self.V_2_att_bias = tf.Variable(np.zeros(1, dtype=np.float32))
            self.V_2_temperature = tf.Variable(np.ones(1, dtype=np.float32))

            self.trainable_variables.extend([
                self.E_2_W, self.E_2_b,
                self.E_2_att, self.E_2_att_bias, self.E_2_temperature,
                self.V_2_W, self.V_2_b,
                self.V_2_att, self.V_2_att_bias, self.V_2_temperature
            ])
        if self.n_layers == 3:
            # dimensionalities
            E_3_W_in = self.hyperedge_type_dim + self.node_dim
            V_3_W_in = self.node_dim + self.hyperedge_dim

            # 2nd layer hyperedges embedding
            self.E_3_att = glorot_normal(shape=(self.node_dim, 1))
            self.E_3_att_bias = tf.Variable(np.zeros(1, dtype=np.float32))
            self.E_3_temperature = tf.Variable(np.ones(1, dtype=np.float32))
            self.E_3_W = glorot_normal(shape=(E_3_W_in, self.hyperedge_dim))
            self.E_3_b = glorot_normal(shape=(1, self.hyperedge_dim))

            # 2nd layer nodes embedding
            self.V_3_W = glorot_normal(shape=(V_3_W_in, self.node_dim))
            self.V_3_b = glorot_normal(shape=(1, self.node_dim))
            self.V_3_att = glorot_normal(shape=(self.hyperedge_dim, 1))
            self.V_3_att_bias = tf.Variable(np.zeros(1, dtype=np.float32))
            self.V_3_temperature = tf.Variable(np.ones(1, dtype=np.float32))

            self.trainable_variables.extend([
                self.E_3_W, self.E_3_b,
                self.E_3_att, self.E_3_att_bias, self.E_3_temperature,
                self.V_3_W, self.V_3_b,
                self.V_3_att, self.V_3_att_bias, self.V_3_temperature
            ])

    def attention(self, seq, att, temperature, bias):
        logits = tf.ragged.map_flat_values(tf.matmul, seq, att)
        logits = temperature * self.attention_activation(logits + bias[0])

        # numerical stability
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)
        ai = tf.math.exp(logits)
        att_weights = ai / tf.math.reduce_sum(ai, axis=1, keepdims=True)
        weighted_input = seq * att_weights
        result = tf.math.reduce_sum(weighted_input, axis=1)
        return result

    def call(self, V, V2E, E2V, hyperedges_type):
        # hyperedges embedding
        E_seq = tf.nn.embedding_lookup(V, E2V)
        # E_pool = self.attention(E_seq, self.E_att,
        #                         self.E_temperature, self.E_att_bias)
        print(E_seq.dtype, E_seq.shape)
        E_pool = self.attention_E_1(E_seq)
        E_type2vec = tf.nn.embedding_lookup(self.E_type, hyperedges_type)
        E_concat = tf.concat([E_pool, E_type2vec], axis=-1)
        E = self.hyperedge_activation(tf.matmul(E_concat, self.E_W) + self.E_b)

        # nodes embedding
        V_seq = tf.nn.embedding_lookup(E, V2E)
        V_pool = self.attention_V_1(V_seq)
        V_concat = tf.concat([V_pool, V], axis=-1)
        V_2 = self.node_activation(tf.matmul(V_concat, self.V_W) + self.V_b)

        if self.n_layers >= 2:
            # 2nd layer hyperedges embedding
            E_2_seq = tf.nn.embedding_lookup(V_2, E2V)
            E_2_pool = self.attention(E_2_seq, self.E_2_att,
                                      self.E_2_temperature, self.E_2_att_bias)
            E_2_concat = tf.concat([E_2_pool, E_type2vec], axis=-1)
            E_2 = self.hyperedge_activation(
                tf.matmul(E_2_concat, self.E_2_W) + self.E_2_b)

            # 2nd layer nodes embedding
            V_2_seq = tf.nn.embedding_lookup(E_2, V2E)
            V_2_pool = self.attention(V_2_seq, self.V_2_att,
                                      self.V_2_temperature, self.V_2_att_bias)
            V_2_concat = tf.concat([V_2_pool, V_2], axis=-1)
            V_3 = self.node_activation(
                tf.matmul(V_2_concat, self.V_2_W) + self.V_2_b)

            if self.n_layers >= 3:
                # 3rd layer hyperedges embedding
                E_3_seq = tf.nn.embedding_lookup(V_3, E2V)
                E_3_pool = self.attention(E_3_seq, self.E_3_att,
                                          self.E_3_temperature,
                                          self.E_3_att_bias)
                E_3_concat = tf.concat([E_3_pool, E_type2vec], axis=-1)
                E_3 = self.hyperedge_activation(
                    tf.matmul(E_3_concat, self.E_3_W) + self.E_3_b)

                # 3rd layer nodes embedding
                V_3_seq = tf.nn.embedding_lookup(E_3, V2E)
                V_3_pool = self.attention(V_3_seq, self.V_3_att,
                                          self.V_3_temperature,
                                          self.V_3_att_bias)
                V_3_concat = tf.concat([V_3_pool, V_3], axis=-1)
                V_4 = self.node_activation(
                    tf.matmul(V_3_concat, self.V_3_W) + self.V_3_b)

                # get label
                out = tf.nn.softmax(tf.matmul(V_4, self.W))
            else:
                # get label
                out = tf.nn.softmax(tf.matmul(V_3, self.W))
        else:
            # get label
            out = tf.nn.softmax(tf.matmul(V_2, self.W))
        return out

    def get_optimizer(
        self, optimizer, learning_rate, beta_1, beta_2, clipnorm,
        momentum=.5, nesterov=False
    ):
        if optimizer == "nadam":
            opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                            beta_1=beta_1,
                                            beta_2=beta_2,
                                            clipnorm=clipnorm)
        elif optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                           beta_1=beta_1,
                                           beta_2=beta_2,
                                           clipnorm=clipnorm)
        elif optimizer == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,
                                              clipnorm=clipnorm)
        elif optimizer == "adamax":
            opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate,
                                             beta_1=beta_1,
                                             beta_2=beta_2,
                                             clipnorm=clipnorm)
        elif optimizer == "adagrad":
            opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer == "sgd":
            opt = tf.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
        return opt

    def fit(
        self, H, V, y, embedding_dim=None,
        preprocessor=None,
        epochs=200, learning_rate=1e-3, optimizer="nadam",
        beta_1=.9, beta_2=.999, clipnorm=2,
        metrics="accuracy", validation_data=None
    ):
        # prepare training set
        n_labels = int(np.max(list(y.values())) + 1)
        y_true = np.zeros((H.n_nodes, n_labels), dtype=np.bool_)
        for node, label in y.items():
            node_id = H.node2id[node]
            y_true[node_id, label] = 1
        # cardinalities
        self.n_labels = n_labels
        self.n_hyperedges_type = H.n_hyperedges_type

        # ensure embedding weights are float32
        if isinstance(V, tf.Variable):
            assert V.dtype == np.float32
            self.trainable_variables.append(V)
        elif V.dtype != np.float32:
            V = V.astype(np.float32)

        # add preprocessor model weights
        if preprocessor is not None:
            V = preprocessor(tf.constant(V, dtype=np.float32))
            self.trainable_variables.extend(preprocessor.trainable_variables)
            self.preprocessor = preprocessor

        # extract embedding size from input
        self.embedding_dim = V.shape[1]

        # create training variables
        self.build()

        # hypergraph variables
        E2V = H.E2V  # hyperedge to vertices list
        V2E = H.V2E  # vertice to hyperedges list
        hyperedges_type = np.array(H.hyperedges_type, dtype=np.int32)

        # get optimizer
        opt = self.get_optimizer(optimizer,
                                 learning_rate,
                                 beta_1, beta_2,
                                 clipnorm)

        best_val_acc = float("-inf")  # used to store validation accuracies
        pbar = tqdm(range(epochs))
        for _ in pbar:
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                output = self.call(V, V2E, E2V, hyperedges_type)
                loss = tf.math.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(y_true, output))

                # print accuracy values
                if metrics == "accuracy":
                    y_pred = np.argmax(output.numpy(), axis=-1)
                    accuracy = 0
                    for node, true_label in y.items():
                        accuracy += true_label == y_pred[H.node2id[node]]
                    accuracy /= len(y)
                    if validation_data is not None:
                        val_accuracy = 0
                        for node, true_label in validation_data.items():
                            val_accuracy += (
                                true_label == y_pred[H.node2id[node]])
                        val_accuracy /= len(validation_data)
                        if val_accuracy > best_val_acc:
                            best_val_acc = val_accuracy
                        pbar.set_description(
                            f"accuracy={accuracy:.3f} "
                            f"- val_acc={val_accuracy:.3f}")
                    else:
                        pbar.set_description(f"accuracy={accuracy:.3f}")
                else:
                    pbar.set_description(f"loss={loss.numpy():.3f}")

                # gradient descent
                gradients = tape.gradient(loss, self.trainable_variables)
                opt.apply_gradients(zip(gradients, self.trainable_variables))

        # show best validation accuracy
        if validation_data is not None:
            print(f"[i] best_val_acc={best_val_acc:.3f}")

    def predict(self, H, V, as_dict=True):
        if V.dtype != np.float32:
            V = V.astype(np.float32)

        if hasattr(self, "preprocessor"):
            V = self.preprocessor.predict(V, verbose=False)

        E2V = H.E2V  # hyperedge to vertices list
        V2E = H.V2E  # vertice to hyperedges list
        hyperedges_type = np.array(H.hyperedges_type, dtype=np.int32)
        output = self.call(V, V2E, E2V, hyperedges_type)
        y_pred = np.argmax(output.numpy(), axis=-1)

        if as_dict:
            res = {H.nodes[i]: label for i, label in enumerate(y_pred)}
            return res
        return y_pred

    def save(self, filename):
        import pickle
        pickle.dump(self, open(filename, "wb"))

    @staticmethod
    def load(filename):
        import pickle
        return pickle.load(open(filename, "rb"))


def split_train_test(y, validation_split=.1, random_state=None):
    r = np.random.RandomState(random_state)

    nodes = list(y.keys())
    if validation_split > 1:
        k = validation_split
    else:
        k = int(round((1-validation_split) * len(nodes)))
    train_set = r.choice(nodes, size=k, replace=False)
    test_set = [n for n in nodes if n not in train_set]

    y_train = {n: y[n] for n in train_set}
    y_test = {n: y[n] for n in test_set}
    return y_train, y_test


def get_hyperedges_from_label_matrix(X):
    hyperedges = []
    for i in range(X.shape[1]):
        he = list(np.where(X[:, i] != 0)[0])
        if len(he) == 0:
            continue
        hyperedges.append(he)
    return hyperedges


def glorot_normal(shape):
    if len(shape) == 1:
        stddev = (1/shape[0])**.5
    else:
        stddev = (2 / (shape[0] + shape[1]))**.5
    return tf.Variable(normal(size=shape, scale=stddev),
                       dtype=np.float32)


def get_weighted_attention(attention_initializer="glorot_uniform"):
    return WeightedAttention(attention_initializer)


def get_multihead_self_attention(in_dim):
    inp = tf.keras.layers.Input(
        shape=(None, in_dim), ragged=True, dtype=np.float32)
    x = tf.keras.layers.MultiHeadAttention(4, 24)(inp, inp, inp)
    # out = WeightedAttention()(x)
    model = tf.keras.Model(inp, x)
    return model


@tf.keras.utils.register_keras_serializable()
class WeightedAttention(tf.keras.layers.Layer):
    def __init__(
            self,
            attention_initializer="glorot_uniform",
            attention_activation=None,
            **kwargs):

        super().__init__(**kwargs)
        self.attention_activation = tf.keras.activations.get(
            attention_activation)
        self.attention_initializer = attention_initializer

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.att = self.add_weight(name="att",
                                   shape=(dim, 1),
                                   initializer=self.attention_initializer)
        self.bias = self.add_weight(name="bias",
                                    shape=(1,),
                                    initializer="zeros")
        self.temperature = self.add_weight(name="temperature",
                                           shape=(1,),
                                           initializer="ones")

    def call(self, seq):
        logits = tf.ragged.map_flat_values(tf.matmul, seq, self.att)
        logits = self.temperature * self.attention_activation(
            logits + self.bias[0])

        # numerical stability
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)
        ai = tf.math.exp(logits)
        att_weights = ai / tf.math.reduce_sum(ai, axis=1, keepdims=True)
        weighted_input = seq * att_weights
        result = tf.math.reduce_sum(weighted_input, axis=1)
        return result

    def get_config(self):
        config = super(WeightedAttention, self).get_config()
        config.update({"hidden_dim": self.hidden_dim,
                       "bias_regularizer": self.bias_regularizer,
                       "attention_initializer": self.attention_initializer,
                       "attention_activation": self.attention_activation,
                       "attention_regularizer": self.attention_regularizer})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
