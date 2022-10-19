import numpy as np
import tensorflow as tf
from numpy.random import normal
from tqdm import tqdm

WALK_LEN = 5
LOUVAIN_RESOLUTION = 1
ROLEWALK_METHOD = "kmeans"


class HyperGraph:
    def __init__(self, G, methods=["neighbors", "louvain"]):
        self.hyperedges_type = []
        self.hyperedges = []
        self.current_hyperedges_type = 0
        self.node2id = {n: i for i, n in enumerate(G.nodes)}
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
        self.n_hyperedges_type = self.current_hyperedges_type

    def add_hyperedges_from(self, hyperedges, ids=True):
        hyperedge_index = len(self.hyperedges)
        for i, hyperedge in enumerate(hyperedges):
            if ids:
                for node_id in hyperedge:
                    self.node2hyperedges[node_id].append(hyperedge_index + i)
                self.hyperedges.append(hyperedge)
            else:
                h = []
                for node in hyperedge:
                    node_id = self.node2id[node]
                    self.node2hyperedges[node_id].append(hyperedge_index + i)
                    h.append(node_id)
                self.hyperedges.append(h)
            self.hyperedges_type.append(self.current_hyperedges_type)
        self.current_hyperedges_type += 1

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

    def add_random_walks_hyperedges(self, G):
        from walker import random_walks
        X = random_walks(G, n_walks=1, walk_len=WALK_LEN)
        hyperedge_index = len(self.hyperedges)
        for i, row in enumerate(X):
            hyperedge = []
            for node_id in row:
                self.node2hyperedges[node_id].append(i + hyperedge_index)
                hyperedge.append(node_id)
            self.hyperedges.append(hyperedge)
            self.hyperedges_type.append(self.current_hyperedges_type)
        self.current_hyperedges_type += 1

    def add_rolewalk_hyperedges(self, G):
        from rolewalk import RoleWalk
        y = RoleWalk(walk_len=3).fit_predict(G, method=ROLEWALK_METHOD)
        hyperedges = {}
        for node_id, cm in enumerate(y):
            hyperedges.setdefault(cm, []).append(node_id)
        print(np.bincount(y))

        hyperedge_index = len(self.hyperedges)
        hyperedges = list(hyperedges.values())
        for i, nodes in enumerate(hyperedges):
            for node_id in nodes:
                self.node2hyperedges[node_id].append(i + hyperedge_index)

        self.hyperedges.extend(hyperedges)
        self.hyperedges_type.extend(
            [self.current_hyperedges_type] * len(hyperedges))
        self.current_hyperedges_type += 1

    def add_louvain_hyperedges(self, G):
        from cylouvain import best_partition
        partition = best_partition(G, resolution=LOUVAIN_RESOLUTION)
        hyperedges = {}
        for node, cm in partition.items():
            node_id = self.node2id[node]
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

    def add_neighbors_of_neighbors_hyperedges(self, G):
        hyperedge_index = len(self.hyperedges)
        for i, node in enumerate(G.nodes):
            node_id = self.node2id[node]
            hyperedge = set()
            # include node itself in hyperedge
            hyperedge.add(node_id)
            self.node2hyperedges[node_id].append(i + hyperedge_index)

            for neighbor in G.neighbors(node):
                for neighbor_2 in G.neighbors(neighbor):
                    nb_id = self.node2id[neighbor_2]
                    hyperedge.add(nb_id)
                    self.node2hyperedges[nb_id].append(i + hyperedge_index)

            self.hyperedges.append(list(hyperedge))
            self.hyperedges_type.append(self.current_hyperedges_type)
        self.current_hyperedges_type += 1

    def add_self_hyperedges(self, G):
        hyperedge_index = len(self.hyperedges)
        for i, node in enumerate(G.nodes):
            node_id = self.node2id[node]
            self.node2hyperedges[node_id].append(i + hyperedge_index)
            self.hyperedges.append([node_id])
            self.hyperedges_type.append(self.current_hyperedges_type)
        self.current_hyperedges_type += 1

    @property
    def V2E(self):
        return tf.ragged.constant(
            [self.node2hyperedges[i] for i in range(self.n_nodes)],
            dtype=np.int32)

    @property
    def E2V(self):
        return tf.ragged.constant(self.hyperedges, dtype=np.int32)


class HyperGNN:
    def __init__(
        self,
        n_labels,
        n_hyperedges_type,
        embedding_dim=10,
        hyperedge_type_dim=5,
        hyperedge_dim=10,
        node_dim=10,
        hyperedge_activation="tanh",
        node_activation="tanh"
    ):
        # cardinalities
        self.n_labels = n_labels
        self.n_hyperedges_type = n_hyperedges_type

        # dimensionality
        self.embedding_dim = embedding_dim
        self.hyperedge_type_dim = hyperedge_type_dim
        self.hyperedge_dim = hyperedge_dim
        self.node_dim = node_dim
        self.training_variables = []

        # activations
        self.hyperedge_activation = (
            tf.keras.activations.get(hyperedge_activation))
        self.node_activation = tf.keras.activations.get(node_activation)

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
        self.E_temperature = tf.Variable(np.ones(1, dtype=np.float32))

        # nodes embedding
        self.V_W = glorot_normal(shape=(V_W_in, self.node_dim))
        self.V_b = glorot_normal(shape=(1, self.node_dim))
        self.V_att = glorot_normal(shape=(self.hyperedge_dim, 1))
        self.V_temperature = tf.Variable(np.ones(1, dtype=np.float32))

        # classification weights
        self.W = glorot_normal(shape=(self.node_dim, self.n_labels))

        # list training variables
        self.training_variables = [
            self.E_type, self.E_W, self.E_b, self.E_att, self.E_temperature,
            self.V_W, self.V_b, self.V_att, self.V_temperature,
            self.W]

    def attention(self, E_seq, E_att, temperature):
        logits = tf.ragged.map_flat_values(tf.matmul, E_seq, E_att)
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)
        ai = tf.math.exp(temperature * logits)
        att_weights = ai / tf.math.reduce_sum(ai, axis=1, keepdims=True)
        weighted_input = E_seq * att_weights
        result = tf.math.reduce_sum(weighted_input, axis=1)
        return result

    def call(self, V, V2E, E2V, hyperedges_type):
        # hyperedges embedding
        E_seq = tf.nn.embedding_lookup(V, E2V)
        E_pool = self.attention(E_seq, self.E_att, self.E_temperature)
        E_type2vec = tf.nn.embedding_lookup(self.E_type, hyperedges_type)
        E_concat = tf.concat([E_pool, E_type2vec], axis=-1)
        E = self.hyperedge_activation(tf.matmul(E_concat, self.E_W) + self.E_b)

        # nodes encoding
        V_seq = tf.nn.embedding_lookup(E, V2E)
        V_pool = self.attention(V_seq, self.V_att, self.V_temperature)
        V_concat = tf.concat([V_pool, V], axis=-1)
        V_2 = self.node_activation(tf.matmul(V_concat, self.V_W) + self.V_b)

        # get label
        out = tf.nn.softmax(tf.matmul(V_2, self.W))
        return out

    def get_optimizer(
        self, optimizer, learning_rate, beta_1, beta_2, clipnorm
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
        return opt

    def fit(
        self, H, y, V=None,
        epochs=200, learning_rate=1e-3, optimizer="nadam",
        beta_1=.9, beta_2=.999, clipnorm=2,
        metrics="accuracy", validation_data=None
    ):
        # create training variables
        self.build()

        # prepare training set
        n_labels = np.max(list(y.values())) + 1
        y_true = np.zeros((H.n_nodes, n_labels))
        for node, label in y.items():
            node_id = H.node2id[node]
            y_true[node_id, label] = 1

        # embedding weights
        if V is None:
            V = glorot_normal(shape=(H.n_nodes, self.embedding_dim))
            self.V = V
            self.training_variables.append(V)
        elif V.dtype != np.float32:
            V = V.astype(np.float32)

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
                tape.watch(self.training_variables)
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
                gradients = tape.gradient(loss, self.training_variables)
                opt.apply_gradients(zip(gradients, self.training_variables))

        # show best validation accuracy
        if validation_data is not None:
            print(f"[i] best_val_acc={best_val_acc:.3f}")

    def predict(self, H, V=None):
        if V is None:
            V = self.V
        elif V.dtype != np.float32:
            V = V.astype(np.float32)

        E2V = H.E2V  # hyperedge to vertices list
        V2E = H.V2E  # vertice to hyperedges list
        hyperedges_type = np.array(H.hyperedges_type, dtype=np.int32)
        output = self.call(V, V2E, E2V, hyperedges_type)
        y_pred = np.argmax(output.numpy(), axis=-1)
        return y_pred

    def save(self, filename):
        import pickle
        pickle.dump(self, open(filename, "wb"))

    @staticmethod
    def load(filename):
        import pickle
        return pickle.load(open(filename, "rb"))


class Adam(object):
    def __init__(
        self, lr=.01, beta_1=.9, beta_2=.999, epsilon=1e-7, clipnorm=None
    ):
        self._lr = tf.Variable(lr, dtype=tf.float32)
        self._beta_1 = tf.Variable(beta_1, dtype=tf.float32)
        self._beta_2 = tf.Variable(beta_2, dtype=tf.float32)
        self._epsilon = tf.constant(epsilon, dtype=tf.float32)
        self.clipnorm = clipnorm
        self._t = tf.Variable(0, dtype=tf.float32)

    def init_moments(self, var_list):
        self._m = {var._unique_id: tf.Variable(tf.zeros_like(var))
                   for var in var_list}
        self._v = {var._unique_id: tf.Variable(tf.zeros_like(var))
                   for var in var_list}

    def apply_gradients(self, grads_and_vars):
        self._t.assign_add(tf.constant(1., self._t.dtype))
        for grad, var in grads_and_vars:
            if self.clipnorm:
                grad = tf.clip_by_norm(grad, self.clipnorm)

            m = self._m[var._unique_id]
            v = self._v[var._unique_id]

            m.assign(self._beta_1 * m + (1. - self._beta_1) * grad)
            v.assign(self._beta_2 * v + (1. - self._beta_2) * tf.square(grad))

            lr = self._lr * \
                tf.sqrt(1 - tf.pow(self._beta_2, self._t)) / \
                (1 - tf.pow(self._beta_1, self._t))
            update = -lr * m / (tf.sqrt(v) + self._epsilon)
            var.assign_add(update)


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
