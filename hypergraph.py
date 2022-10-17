import tensorflow as tf
import numpy as np
# from optimizer import Adam
from tqdm import tqdm
from numpy.random import normal


class HyperGraph:
    def __init__(self, G, methods=["self"]):
        self.hyperedges_type = []
        self.hyperedges = []
        self.current_hyperedges_type = 0
        self.node2id = {n: i for i, n in enumerate(G.nodes)}
        self.node2hyperedges = [[] for i in range(len(G.nodes))]
        self.n_nodes = len(G.nodes)
        
        if "louvain" in methods:
            self.add_louvain_hyperedges(G)
        if "neighbors" in methods:
            self.add_neighbors_hyperedges(G)
        if "self" in methods:
            self.add_self_hyperedges(G)
        self.n_hyperedges_type = self.current_hyperedges_type

    def add_louvain_hyperedges(self, G):
        from cylouvain import best_partition
        partition = best_partition(G)
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
            # hyperedge.append(node_id)
            # self.node2hyperedges[node_id].append(i + hyperedge_index)
            for neighbor in G.neighbors(node):
                nb_id = self.node2id[neighbor]
                if nb_id == node_id:
                    continue
                hyperedge.append(nb_id)
                self.node2hyperedges[node_id].append(i + hyperedge_index)

            self.hyperedges.append(hyperedge)
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
            [self.node2hyperedges[i] for i in range(self.n_nodes)], dtype=np.int32)

    @property
    def E2V(self):
        return tf.ragged.constant(self.hyperedges, dtype=np.int32)


class HGNN:
    def __init__(
        self,
        n_labels,
        n_hyperedges_type,
        embedding_dim=10,
        hyperedge_type_dim=5,
        hyperedge_dim=10,
        node_dim=10,
        hyperedge_activation="tanh",
        node_activation="tanh",
        pooling="mean"
    ):
        # cardinalities
        self.n_labels = n_labels
        self.n_hyperedges_type = n_hyperedges_type

        # dimensionality
        self.embedding_dim = embedding_dim
        self.hyperedge_type_dim = hyperedge_type_dim
        self.hyperedge_dim = hyperedge_dim
        self.node_dim = node_dim
        
        if pooling == "mean":
            self.pooling = lambda x: tf.math.reduce_mean(x, axis=1, keepdims=False)
        elif pooling == "max":
            self.pooling = lambda x: tf.math.reduce_max(x, axis=1, keepdims=False)
        
        # activations
        self.hyperedge_activation = tf.keras.activations.get(hyperedge_activation)
        self.node_activation = tf.keras.activations.get(node_activation)
        self.init()

    def init(self):
        # dimensionalities
        E_W_in = self.hyperedge_type_dim + self.embedding_dim
        V_W_in = self.embedding_dim + self.hyperedge_dim
        
        # hyperedges type embedding
        E_type = normal(
            size=(self.n_hyperedges_type, self.hyperedge_type_dim),
            scale=1/self.hyperedge_type_dim)
        self.E_type = tf.Variable(E_type, dtype=np.float32)
        self.E_W = tf.Variable(normal(size=(E_W_in, self.hyperedge_dim)),
                               dtype=np.float32)
        self.E_b = tf.Variable(normal(size=(1, self.hyperedge_dim)),
                               dtype=np.float32)

        # classification weights
        self.W = tf.Variable(normal(size=(self.node_dim, self.n_labels)),
                             dtype=np.float32)  # trainable
        self.V_W = tf.Variable(normal(size=(V_W_in, self.node_dim)),
                               dtype=np.float32)
        self.V_b = tf.Variable(normal(size=(1, self.node_dim)),
                               dtype=np.float32)

        # list training variables
        self.training_variables = [
            self.E_type, self.E_W, self.E_b,
            self.W, self.V_W, self.V_b]
    
    def call(self, V, V2E, E2V, hyperedges_type):
        # hyperedges embedding
        E_listing = tf.nn.embedding_lookup(V, E2V)
        E_pool = self.pooling(E_listing)
        E_type2vec = tf.nn.embedding_lookup(self.E_type, hyperedges_type)
        E_concat = tf.concat([E_pool, E_type2vec], axis=-1)
        E = self.hyperedge_activation(tf.matmul(E_concat, self.E_W) + self.E_b)

        # nodes encoding
        V_listing = tf.nn.embedding_lookup(E, V2E)
        V_pool = self.pooling(V_listing)
        V_concat = tf.concat([V_pool, V], axis=-1)
        V_2 = self.node_activation(tf.matmul(V_concat, self.V_W) + self.V_b)
        
        # get label
        out = tf.nn.softmax(tf.matmul(V_2, self.W))
        return out

    def fit(
        self, H, y, V=None,
        epochs=200, learning_rate=1e-1,
        metrics="accuracy", validation_data=None
    ):
        # prepare training set
        n_labels = np.max(list(y.values())) + 1
        y_true = np.zeros((H.n_nodes, n_labels))
        for node, label in y.items():
            node_id = H.node2id[node]
            y_true[node_id, label] = 1

        # embedding weights
        if V is None:
            V = normal(size=(H.n_nodes, self.embedding_dim),
                       scale=1/self.embedding_dim)
            V = tf.Variable(V, dtype=np.float32)  # trainable
            self.V = V
            self.training_variables.append(V)
        elif V.dtype != np.float32:
            V = V.astype(np.float32)

        E2V = H.E2V  # hyperedge to vertices list        
        V2E = H.V2E  # vertice to hyperedges list
        hyperedges_type = np.array(H.hyperedges_type, dtype=np.int32)

        optimizer = Adam(lr=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-8, clipnorm=2)
        optimizer.init_moments(self.training_variables)
        
        pbar = tqdm(range(epochs))
        for epoch in pbar:
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
                    pbar.set_description(f"accuracy={accuracy:.3f}")
                else:
                    pbar.set_description(f"loss={loss.numpy():.3f}")

                gradients = tape.gradient(loss, self.training_variables)
                optimizer.apply_gradients(zip(gradients, self.training_variables))
            if validation_data is not None and epoch == epochs - 1:
                y_pred = np.argmax(output.numpy(), axis=-1)
                val_accuracy = 0
                for node, true_label in validation_data.items():
                    val_accuracy += true_label == y_pred[H.node2id[node]]
                val_accuracy /= len(validation_data)
                pbar.set_description(f"accuracy={accuracy:.3f} "
                                     f"- val_accuracy={val_accuracy:.2f}")

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
        

class Adam(object):
    def __init__(self, lr=.01, beta_1=.9, beta_2=.999, epsilon=1e-8, clipnorm=None):
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
            if self.clipnorm: grad = tf.clip_by_norm(grad, self.clipnorm)

            m = self._m[var._unique_id]
            v = self._v[var._unique_id]

            m.assign(self._beta_1 * m + (1. - self._beta_1) * grad)
            v.assign(self._beta_2 * v + (1. - self._beta_2) * tf.square(grad))

            lr = self._lr * tf.sqrt(1 - tf.pow(self._beta_2, self._t)) / (1 - tf.pow(self._beta_1, self._t))
            update = -lr * m / (tf.sqrt(v) + self._epsilon)
            var.assign_add(update)


def split_train_test(y, validation_split=.1, random_state=None):
    r = np.random.RandomState(random_state)

    nodes = list(y.keys())
    k = int(round((1-validation_split) * len(nodes)))
    train_set = r.choice(nodes, size=k, replace=False)
    test_set = [n for n in nodes if n not in train_set]

    y_train = {n: y[n] for n in train_set}
    y_test = {n: y[n] for n in test_set}
    return y_train, y_test
