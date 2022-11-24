import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from .utils import (glorot_normal, glorot_uniform, iter_batch, ragged_dot,
                    to_categorical)


class HyperGNN:
    def __init__(
        self,
        hyperedge_type_dim=64,
        hyperedge_dim=256,
        node_dim=256,
        attention_dim=128,
        hyperedge_activation="tanh",
        node_activation="tanh",
        pooling="key_query",
        n_layers=2,
        selfless=False
    ):
        self.hyperedge_type_dim = hyperedge_type_dim
        self.hyperedge_dim = hyperedge_dim
        self.attention_dim = attention_dim
        self.node_dim = node_dim
        self.n_layers = n_layers
        self.selfless = selfless
        self.pooling = pooling

        # activations
        self.hyperedge_activation = (
            tf.keras.activations.get(hyperedge_activation))
        self.node_activation = tf.keras.activations.get(node_activation)

    def add_weight(self, shape, initializer="glorot_normal", name=None):
        if initializer == "glorot_normal":
            parameter = glorot_normal(shape=shape, name=name)
        elif initializer == "glorot_uniform":
            parameter = glorot_uniform(shape=shape, name=name)
        elif initializer == "ones":
            parameter = tf.Variable(
                np.ones(shape, dtype=np.float32), name=name)
        elif initializer == "zeros":
            parameter = tf.Variable(
                np.zeros(shape, dtype=np.float32),
                name=name)

        self.trainable_variables.append(parameter)
        return parameter

    def add_attention(self, attention_type="new"):
        if attention_type == "key_query":
            from .pooling import KeyQueryAttention
            att = KeyQueryAttention(
                latent_dim=self.attention_dim)
        elif attention_type == "weighted":
            from .pooling import WeightedAttention
            att = WeightedAttention()
        elif attention_type == "apln":
            from .pooling import APLN
            att = APLN()
        elif attention_type == "average":
            att = layers.GlobalAveragePooling1D()
        elif attention_type == "max":
            att = layers.GlobalMaxPooling1D()
        elif attention_type == "selective":
            from .pooling import SelectiveAttention
            att = SelectiveAttention(
                latent_dim=self.attention_dim)
        self.trainable_variables.extend(att.trainable_variables)
        return att

    def build_model(self):
        self.trainable_variables = []

        # dimensionalities
        if self.selfless:
            E_W_in = self.hyperedge_type_dim + self.embedding_dim
            V_W_in = self.hyperedge_dim
        else:
            E_W_in = self.hyperedge_type_dim + self.embedding_dim
            V_W_in = self.embedding_dim + self.hyperedge_dim

        # hyperedges embedding
        self.E_type = self.add_weight(shape=(self.n_hyperedges_type,
                                             self.hyperedge_type_dim),
                                      initializer="glorot_uniform",
                                      name="E_type")
        self.E_W = self.add_weight(shape=(E_W_in, self.hyperedge_dim),
                                   name="E_W")
        self.E_b = self.add_weight(shape=(1, self.hyperedge_dim),
                                   initializer="zeros",
                                   name="E_b")
        self.E_type_modulator = self.add_weight(
            shape=(self.n_hyperedges_type, 1, self.attention_dim))

        # nodes embedding
        self.V_W = self.add_weight(shape=(V_W_in, self.node_dim),
                                   name="V_W")
        self.V_b = self.add_weight(shape=(1, self.node_dim),
                                   initializer="zeros",
                                   name="V_b")
        self.V_type_modulator = self.add_weight(
            shape=(self.hyperedge_dim, self.attention_dim),
            name="V_type_modulator")

        # classification weights
        self.W = self.add_weight(shape=(self.node_dim, self.n_labels),
                                 name="W")
        # attention layers
        self.attention_E_1 = self.add_attention("selective")
        self.attention_V_1 = self.add_attention("selective")

        if self.n_layers >= 2:
            # dimensionalities
            E_2_W_in = self.hyperedge_type_dim + self.node_dim
            V_2_W_in = self.hyperedge_dim + self.node_dim

            # 2nd layer hyperedges embedding
            self.E_2_W = self.add_weight(shape=(E_2_W_in, self.hyperedge_dim),
                                         name="E_2_W")
            self.E_2_b = self.add_weight(shape=(1, self.hyperedge_dim),
                                         initializer="zeros",
                                         name="E_2_b")
            self.E_2_type_modulator = self.add_weight(
                shape=(self.n_hyperedges_type, 1, self.attention_dim),
                name="E_2_type_modulator")

            # 2nd layer nodes embedding
            self.V_2_W = self.add_weight(shape=(V_2_W_in, self.node_dim),
                                         name="V_2_W")
            self.V_2_b = self.add_weight(shape=(1, self.node_dim),
                                         name="V_2_b")
            self.V_2_type_modulator = self.add_weight(
                shape=(self.hyperedge_dim, self.attention_dim),
                name="V_2_type_modulator")

            self.attention_E_2 = self.add_attention("selective")
            self.attention_V_2 = self.add_attention("selective")
        if self.n_layers == 3:
            # dimensionalities
            E_3_W_in = self.hyperedge_type_dim + self.node_dim
            V_3_W_in = self.node_dim + self.hyperedge_dim

            # 2nd layer hyperedges embedding
            self.E_3_W = self.add_weight(shape=(E_3_W_in, self.hyperedge_dim))
            self.E_3_b = self.add_weight(shape=(1, self.hyperedge_dim),
                                         initializer="zeros")

            # 2nd layer nodes embedding
            self.V_3_W = self.add_weight(shape=(V_3_W_in, self.node_dim))
            self.V_3_b = self.add_weight(shape=(1, self.node_dim),
                                         initializer="zeros")
            self.attention_E_3 = self.add_attention(self.pooling)
            self.attention_V_3 = self.add_attention(self.pooling)

    def call(self, V, V2E, E2V, hyperedges_type, return_node_embeddings=False):
        # hyperedges embedding
        E_type2vec = tf.nn.embedding_lookup(self.E_type, hyperedges_type)
        E_seq = tf.nn.embedding_lookup(V, E2V)
        E_att_modulator = tf.nn.embedding_lookup(self.E_type_modulator,
                                                 hyperedges_type)
        E_pool = self.attention_E_1(E_seq, E_att_modulator)
        E_concat = tf.concat([E_pool, E_type2vec], axis=-1)
        E = self.hyperedge_activation(tf.matmul(E_concat, self.E_W) + self.E_b)

        # nodes embedding
        V_seq = tf.nn.embedding_lookup(E, V2E)
        V_att_modulator = ragged_dot(V_seq, self.V_type_modulator)
        V_pool = self.attention_V_1(V_seq, V_att_modulator)
        V_concat = tf.concat([V_pool, V], axis=-1)
        V_2 = self.node_activation(tf.matmul(V_concat, self.V_W) + self.V_b)

        # return embeddings if needed
        if return_node_embeddings and self.n_layers == 1:
            return V_2

        if self.n_layers >= 2:
            # 2nd layer hyperedges embedding
            E_2_seq = tf.nn.embedding_lookup(V_2, E2V)
            E_2_att_modulator = tf.nn.embedding_lookup(self.E_2_type_modulator,
                                                       hyperedges_type)
            E_2_pool = self.attention_E_2(E_2_seq, E_2_att_modulator)
            E_2_concat = tf.concat([E_2_pool, E_type2vec], axis=-1)
            E_2 = self.hyperedge_activation(
                tf.matmul(E_2_concat, self.E_2_W) + self.E_2_b)

            # 2nd layer nodes embedding
            V_2_seq = tf.nn.embedding_lookup(E_2, V2E)
            V_2_att_modulator = ragged_dot(V_seq, self.V_2_type_modulator)
            V_2_pool = self.attention_V_2(V_2_seq, V_2_att_modulator)
            if self.selfless:
                V_2_concat = V_2_pool
            else:
                V_2_concat = tf.concat([V_2_pool, V_2], axis=-1)
            V_3 = self.node_activation(
                tf.matmul(V_2_concat, self.V_2_W) + self.V_2_b)
            if return_node_embeddings and self.n_layers == 2:
                return V_3

            if self.n_layers >= 3:
                # 3rd layer hyperedges embedding
                E_3_seq = tf.nn.embedding_lookup(V_3, E2V)
                E_3_pool = self.attention_E_3(E_3_seq)
                E_3_concat = tf.concat([E_3_pool, E_type2vec], axis=-1)
                E_3 = self.hyperedge_activation(
                    tf.matmul(E_3_concat, self.E_3_W) + self.E_3_b)

                # 3rd layer nodes embedding
                V_3_seq = tf.nn.embedding_lookup(E_3, V2E)
                V_3_pool = self.attention_V_3(V_3_seq)
                V_3_concat = tf.concat([V_3_pool, V_3], axis=-1)
                V_4 = self.node_activation(
                    tf.matmul(V_3_concat, self.V_3_W) + self.V_3_b)
                if return_node_embeddings:
                    return V_4

                # get label
                out = tf.nn.softmax(tf.matmul(V_4, self.W))
            else:
                # get label
                out = tf.nn.softmax(tf.matmul(V_3, self.W))
        else:
            # get label
            out = tf.nn.softmax(tf.matmul(V_2, self.W))
        return out

    def call_on_batch(
        self, V, V2E, E2V, hyperedges_type,
        return_node_embeddings=False,
        nodes_slice=None, nodes_restricted=None
    ):
        V_r = V[nodes_restricted]

        # hyperedges embedding
        E_type2vec = tf.nn.embedding_lookup(self.E_type, hyperedges_type)
        E_seq = tf.nn.embedding_lookup(V_r, E2V)
        E_att_modulator = tf.nn.embedding_lookup(self.E_type_modulator,
                                                 hyperedges_type)
        E_pool = self.attention_E_1(E_seq, E_att_modulator)
        E_concat = tf.concat([E_pool, E_type2vec], axis=-1)
        E = self.hyperedge_activation(tf.matmul(E_concat, self.E_W) + self.E_b)

        # nodes embedding
        V_seq = tf.nn.embedding_lookup(E, V2E)
        V_att_modulator = ragged_dot(V_seq, self.V_type_modulator)
        V_pool = self.attention_V_1(V_seq, V_att_modulator)
        V_concat = tf.concat([V_pool, V_r], axis=-1)
        V_2 = self.node_activation(tf.matmul(V_concat, self.V_W) + self.V_b)
        V_2_slice = tf.nn.embedding_lookup(V_2, nodes_slice)

        # return embeddings if needed
        if return_node_embeddings and self.n_layers == 1:
            return V_2_slice

        if self.n_layers >= 2:
            # 2nd layer hyperedges embedding
            E_2_seq = tf.nn.embedding_lookup(V_2, E2V)
            E_2_att_modulator = tf.nn.embedding_lookup(self.E_2_type_modulator,
                                                       hyperedges_type)
            E_2_pool = self.attention_E_2(E_2_seq, E_2_att_modulator)
            E_2_concat = tf.concat([E_2_pool, E_type2vec], axis=-1)
            E_2 = self.hyperedge_activation(
                tf.matmul(E_2_concat, self.E_2_W) + self.E_2_b)

            # 2nd layer nodes embedding
            V_2_seq = tf.nn.embedding_lookup(E_2, V2E)
            V_2_att_modulator = ragged_dot(V_seq, self.V_2_type_modulator)
            V_2_pool = self.attention_V_2(V_2_seq, V_2_att_modulator)
            V_2_concat = tf.concat([V_2_pool, V_2], axis=-1)
            V_3 = self.node_activation(
                tf.matmul(V_2_concat, self.V_2_W) + self.V_2_b)
            V_3_slice = tf.nn.embedding_lookup(V_3, nodes_slice)
            if return_node_embeddings and self.n_layers == 2:
                return V_3_slice

            if self.n_layers >= 3:
                # 3rd layer hyperedges embedding
                E_3_seq = tf.nn.embedding_lookup(V_3, E2V)
                E_3_pool = self.attention_E_3(E_3_seq)
                E_3_concat = tf.concat([E_3_pool, E_type2vec], axis=-1)
                E_3 = self.hyperedge_activation(
                    tf.matmul(E_3_concat, self.E_3_W) + self.E_3_b)

                # 3rd layer nodes embedding
                V_3_seq = tf.nn.embedding_lookup(E_3, V2E)
                V_3_pool = self.attention_V_3(V_3_seq)
                V_3_concat = tf.concat([V_3_pool, V_3], axis=-1)
                V_4 = self.node_activation(
                    tf.matmul(V_3_concat, self.V_3_W) + self.V_3_b)
                if return_node_embeddings:
                    return V_4

                # get label
                out = tf.nn.softmax(tf.matmul(V_4, self.W))
            else:
                # get label
                out = tf.nn.softmax(tf.matmul(V_3_slice, self.W))
        else:
            # get label
            out = tf.nn.softmax(tf.matmul(V_2_slice, self.W))
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

    def prepare_training_set(self, H, y):
        n_labels = int(np.max(list(y.values())) + 1)
        y_true = np.zeros((H.n_nodes, n_labels), dtype=np.bool_)
        for node, label in y.items():
            node_id = H.node2id[node]
            y_true[node_id, label] = 1
        # cardinalities
        self.n_labels = n_labels
        self.n_hyperedges_type = H.n_hyperedges_type
        return y_true

    def prepare_features(self, V, preprocessor):
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
        return V

    def fit(
        self, H, V, y,
        preprocessor=None, batch_size=None,
        epochs=200, learning_rate=1e-3, optimizer="nadam",
        beta_1=.9, beta_2=.999, clipnorm=2,
        metrics="accuracy", validation_data=None
    ):
        self.trainable_variables = []

        # prepare training set and features
        y_true = self.prepare_training_set(H, y)
        V = self.prepare_features(V, preprocessor)

        # create training variables
        self.build_model()

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
            if batch_size is None:
                with tf.GradientTape() as tape:
                    tape.watch(self.trainable_variables)
                    output = self.call(V, V2E, E2V, hyperedges_type)
                    loss = tf.math.reduce_mean(
                        tf.keras.losses.categorical_crossentropy(
                            y_true, output))
                    # update tqdm bar
                    best_val_acc = self.update_metrics(
                        pbar, H, y, output, loss,
                        metrics, validation_data, best_val_acc)

                    # gradient descent
                    gradients = tape.gradient(loss, self.trainable_variables)
                    opt.apply_gradients(
                        zip(gradients, self.trainable_variables))
            else:
                for (
                    batch_nodes, nodes_slice, nodes_list,
                    V2E_batch, E2V_batch, ht_batch
                ) in iter_batch(H, y, batch_size, self.n_layers):

                    with tf.GradientTape() as tape:
                        tape.watch(self.trainable_variables)
                        output = self.call_on_batch(
                            V, V2E_batch, E2V_batch, ht_batch,
                            nodes_slice=nodes_slice,
                            nodes_restricted=nodes_list)
                        y_batch = to_categorical(
                            [y[n] for n in batch_nodes],
                            n_labels=self.n_labels)
                        loss = tf.math.reduce_mean(
                            tf.keras.losses.categorical_crossentropy(
                                y_batch, output))
                        # gradient descent
                        gradients = tape.gradient(
                            loss, self.trainable_variables)
                        opt.apply_gradients(
                            zip(gradients, self.trainable_variables))

                # update tqdm bar
                output = self.call(V, V2E, E2V, hyperedges_type)
                loss = tf.math.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(
                        y_true, output))
                best_val_acc = self.update_metrics(
                    pbar, H, y, output, loss,
                    metrics, validation_data, best_val_acc)

        # show best validation accuracy
        if validation_data is not None:
            print(f"[i] best_val_acc={best_val_acc:.3f}")

    def update_metrics(
        self, pbar, H, y, output, loss, metrics, validation_data, best_val_acc
    ):
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
        return best_val_acc

    def fit_transform(
        self, H, V,
        preprocessor=None,
        epochs=200, learning_rate=1e-3, optimizer="nadam",
        beta_1=.9, beta_2=.999, clipnorm=2,
        metrics="accuracy"
    ):
        self.trainable_variables = []

        y = {node: i for i, node in enumerate(H.nodes)}
        self.fit(H, V, y,
                 preprocessor=preprocessor,
                 epochs=epochs,
                 learning_rate=learning_rate,
                 optimizer=optimizer,
                 beta_1=beta_1,
                 beta_2=beta_2,
                 clipnorm=clipnorm,
                 metrics=metrics,
                 validation_data=None)
        return self.transform(H, V)

    def transform(self, H, V):
        return self.get_node_embedding(H, V)

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

    def get_node_embedding(self, H, V):
        if V.dtype != np.float32:
            V = V.astype(np.float32)

        if hasattr(self, "preprocessor"):
            V = self.preprocessor.predict(V, verbose=False)

        E2V = H.E2V  # hyperedge to vertices list
        V2E = H.V2E  # vertice to hyperedges list
        hyperedges_type = np.array(H.hyperedges_type, dtype=np.int32)
        X = self.call(V, V2E, E2V, hyperedges_type,
                      return_node_embeddings=True)
        return X

    def save(self, filename):
        import pickle
        pickle.dump(self, open(filename, "wb"))

    @staticmethod
    def load(filename):
        import pickle
        return pickle.load(open(filename, "rb"))
