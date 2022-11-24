import numpy as np
import tensorflow as tf
from numpy.random import normal


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


def iter_batch(H, y, batch_size=5, n_layers=1):
    keys = list(y.keys())
    np.random.shuffle(keys)
    n_batches = len(keys) // batch_size + int(len(keys) % batch_size > 0)
    for i in range(n_batches):
        batch = keys[i*batch_size:(i + 1)*batch_size]
        yield H.get_batch(batch, n_layers=n_layers)


def get_hyperedges_from_label_matrix(X):
    hyperedges = []
    for i in range(X.shape[1]):
        he = list(np.where(X[:, i] != 0)[0])
        if len(he) == 0:
            continue
        hyperedges.append(he)
    return hyperedges


def glorot_normal(shape, name):
    if len(shape) == 1:
        stddev = (1/shape[0])**.5
    else:
        stddev = (2 / (shape[0] + shape[1]))**.5
    return tf.Variable(normal(size=shape, scale=stddev),
                       name=name, dtype=np.float32)


def glorot_uniform(shape, name):
    if len(shape) == 1:
        scale = (1/shape[0])**.5
    else:
        scale = (2 / (shape[0] + shape[1]))**.5
    weights = np.random.uniform(-scale, scale, size=shape)
    return tf.Variable(weights, name=name, dtype=np.float32)


def ragged_dot(a, b):
    return tf.ragged.map_flat_values(tf.matmul, a, b)


def to_categorical(labels, n_labels):
    y_true = np.zeros((len(labels), n_labels), dtype=np.bool_)
    for i, j in enumerate(labels):
        y_true[i, j] = 1
    return y_true
