import tensorflow as tf

from .utils import ragged_dot


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
                                    shape=(dim,),
                                    initializer="zeros")
        self.p = self.add_weight(name="p",
                                 shape=(1,),
                                 initializer="ones")

    def call(self, seq, *_):
        logits = ragged_dot(seq, self.att)

        # numerical stability
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)
        ai = tf.math.exp(logits)
        att_weights = ai / tf.math.reduce_sum(ai, axis=1, keepdims=True)
        weighted_input = (seq) * att_weights
        result = tf.math.reduce_sum(weighted_input, axis=1)
        return result


class APLN(tf.keras.layers.Layer):
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
        self.p = self.add_weight(name="p",
                                 shape=(1,),
                                 initializer="ones")

    def call(self, seq):
        logits = ragged_dot(seq, self.att)

        # numerical stability
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)
        ai = tf.math.exp(logits)
        ai **= self.p
        att_weights = ai / tf.math.reduce_sum(ai, axis=1, keepdims=True)
        weighted_input = (seq**self.p) * att_weights
        result = tf.math.reduce_sum(weighted_input, axis=1)**(1/self.p)
        return result


class KeyQueryAttention(tf.keras.layers.Layer):
    def __init__(
            self,
            latent_dim=128,
            attention_initializer="glorot_uniform",
            attention_activation=None,
            **kwargs):

        super().__init__(**kwargs)
        self.attention_activation = tf.keras.activations.get(
            attention_activation)
        self.attention_initializer = attention_initializer
        self.latent_dim = latent_dim

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.key = self.add_weight(name="key",
                                   shape=(dim, self.latent_dim),
                                   initializer=self.attention_initializer)
        self.query = self.add_weight(name="query",
                                     shape=(dim, self.latent_dim),
                                     initializer=self.attention_initializer)
        self.bias = self.add_weight(name="bias",
                                    shape=(dim,),
                                    initializer="zeros")

    def call(self, seq, *_):
        keys = ragged_dot(seq, self.key)
        queries = ragged_dot(seq, self.query)

        logits = tf.reduce_sum(keys * queries, axis=-1, keepdims=True)
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)

        ai = tf.exp(logits)
        att_weights = ai / tf.math.reduce_sum(ai, axis=1, keepdims=True)

        # numerical stability
        weighted_input = (seq + self.bias) * att_weights
        result = tf.math.reduce_sum(weighted_input, axis=1)
        return result


class SelectiveAttention(tf.keras.layers.Layer):
    def __init__(
            self,
            latent_dim=128,
            attention_initializer="glorot_uniform",
            attention_activation=None,
            **kwargs):

        super().__init__(**kwargs)
        self.attention_activation = tf.keras.activations.get(
            attention_activation)
        self.attention_initializer = attention_initializer
        self.latent_dim = latent_dim

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.key = self.add_weight(name="key",
                                   shape=(dim, self.latent_dim),
                                   initializer=self.attention_initializer)
        self.query = self.add_weight(name="query",
                                     shape=(dim, self.latent_dim),
                                     initializer=self.attention_initializer)
        self.bias = self.add_weight(name="bias",
                                    shape=(2,),
                                    initializer="zeros")

    def call(self, seq, modulator):
        keys = tf.nn.tanh(ragged_dot(seq, self.key) + self.bias[0])
        queries = tf.nn.tanh(ragged_dot(seq, self.query) + self.bias[1])
        queries *= modulator

        logits = tf.reduce_sum(keys * queries, axis=-1, keepdims=True)
        logits -= tf.reduce_max(logits, axis=1, keepdims=True)

        ai = tf.exp(logits)
        att_weights = ai / tf.math.reduce_sum(ai, axis=1, keepdims=True)

        # numerical stability
        weighted_input = seq * att_weights
        result = tf.math.reduce_sum(weighted_input, axis=1)
        return result
