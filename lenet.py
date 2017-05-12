import tensorflow as tf
from tensorflow.contrib.layers import flatten


def LeNet(x, out_shape, dropout_keep_prob):
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    with tf.variable_scope("conv_one"):
        conv_one = conv2d(x, patch_shape=(5, 5, 3), n_out_features=6)
    conv_one = simple_assert(conv_one.get_shape()[1:] == (28, 28, 6), conv_one)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    pooling_one = tf.nn.max_pool(conv_one, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pooling_one = simple_assert(pooling_one.get_shape()[1:] == (14, 14, 6), pooling_one)

    # Layer 2: Convolutional. Output = 10x10x16.
    with tf.variable_scope("conv_two"):
        conv_two = conv2d(pooling_one, patch_shape=(5, 5, 6), n_out_features=16)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pooling_two = tf.nn.max_pool(conv_two, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pooling_two = simple_assert(pooling_two.get_shape()[1:] == (5, 5, 16), pooling_two)

    # Flatten. Input = 5x5x16. Output = 400.
    flat = flatten(pooling_two)

    # Dense Layers
    with tf.variable_scope("dense_one"):
        dense_one = dense(flat, in_shape=400, out_shape=120, dropout=dropout_keep_prob)
    with tf.variable_scope("dense_two"):
        dense_two = dense(dense_one, in_shape=120, out_shape=84, dropout=dropout_keep_prob)
    logits = dense(dense_two, in_shape=84, out_shape=out_shape, activation=None)
    return logits


def conv2d(x, patch_shape, n_out_features, stride=1, activation=tf.nn.relu):
    """ Shorthand for creating a convolutional layer """
    mu = 0
    sigma = 0.1
    height, width, depth = patch_shape
    weights = tf.get_variable("weights", (height, width, depth, n_out_features), initializer=tf.truncated_normal_initializer(mu, sigma))
    bias = tf.get_variable("bias", initializer=tf.zeros_initializer(n_out_features))
    without_activation = tf.nn.bias_add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], "VALID"), bias)
    return maybe_activate(without_activation, activation)


def dense(x, in_shape, out_shape, dropout=None, activation=tf.nn.relu):
    """ Shorthand for creating a fully connected or dense layer """
    mu = 0
    sigma = 0.1
    weights = tf.get_variable("weights", (in_shape, out_shape), initializer=tf.truncated_normal_initializer(mu, sigma))
    bias = tf.get_variable("bias", initializer=tf.zeros_initializer(out_shape))
    if dropout is not None:
        x = tf.nn.dropout(x, dropout)
    without_activation = tf.matmul(x, weights) + bias
    return maybe_activate(without_activation, activation)


def maybe_activate(without_activation, activation):
    if activation is not None:
        return activation(without_activation)
    return without_activation


def simple_assert(check: bool, dependency_node):
    assert_op = tf.Assert(check, [dependency_node])
    with tf.control_dependencies([assert_op]):
        return tf.identity(dependency_node)


def print_example(conv_one):
    return tf.Print(conv_one, [conv_one])
