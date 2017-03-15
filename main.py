import os

import tensorflow as tf
import pickle
from sklearn.utils import shuffle

from architecture import LeNet

DATA_DIR = "traffic-signs-data"

BATCH_SIZE = 64
LEARNING_RATE = 0.001
SAMPLE_SIZE = 10000
EPOCHS = 3


def load_data(pickle_path: str):
    full_path = os.path.join(DATA_DIR, pickle_path)
    with open(full_path, mode='rb') as f:
        data = pickle.load(f)
    return data['features'], data['labels']


def main():
    X_train, y_train = load_data('train.p')

    # OK LETS SAMPLE THE DATA
    X_train, y_train = X_train[:SAMPLE_SIZE], y_train[:SAMPLE_SIZE]

    X_valid, y_valid = load_data('valid.p')
    X_test, y_test = load_data('test.p')

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, None)
    one_hot_y = tf.one_hot(y, 43)

    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    training_operation = optimizer.minimize(loss_operation)
    correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = sess.run(accuracy_operation, {x: X_train, y: y_train})
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        print("Model saved")


if __name__ == '__main__':
    main()
