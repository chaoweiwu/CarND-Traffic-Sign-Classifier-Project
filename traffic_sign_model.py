import os

import tensorflow as tf
import pickle
from sklearn.utils import shuffle

import numpy as np
from lenet import LeNet

DATA_DIR = "traffic-signs-data"

BATCH_SIZE = 64
LEARNING_RATE = 0.001
SAMPLE_SIZE = 1000000
EPOCHS = 20

MODELS_DIR = "models"
CURRENT_MODEL_NAME = "lr_0001_batchsize_64"


def load_data(pickle_path: str):
    full_path = os.path.join(DATA_DIR, pickle_path)
    with open(full_path, mode='rb') as f:
        data = pickle.load(f)
    return data['features'], data['labels']


def main():
    # train_model()
    test_accuracy()


def train_model():
    """ Train model and test on validation data """
    X_train, y_train = load_data('train.p')
    n_classes = len(np.unique(y_train))

    X_train = normalize_multichannel(X_train)

    # OK LETS SAMPLE THE DATA
    X_train, y_train = X_train[:SAMPLE_SIZE], y_train[:SAMPLE_SIZE]
    print("Using {} rows".format(X_train.shape[0]))

    X_valid, y_valid = load_data('valid.p')
    X_valid = normalize_multichannel(X_valid)
    X_test, y_test = load_data('test.p')

    with tf.Session() as sess:
        model = TrafficSignModel(sess)
        model.train(X_train, y_train, X_valid, y_valid, epochs=2)


def test_accuracy():
    """ Run final test to see model performance """
    X_train, y_train = load_data('train.p')
    X_valid, y_valid = load_data('valid.p')
    X_test, y_test = load_data('test.p')

    # Join test and valid so we have more training data for our final prediction accuracy.
    X_train = np.concatenate((X_train, X_valid))
    y_train = np.concatenate((y_train, y_valid))
    normalize_multichannel(X_train)
    normalize_multichannel(X_test)

    with tf.Session() as sess:
        model = TrafficSignModel(sess)
        model.train(X_train, y_train, X_valid, y_valid, epochs=20)
        model.accuracy(X_test, y_test)


class TrafficSignModel:
    def __init__(self, session: tf.Session):
        self.sess = session
        n_output_classes = 43
        with tf.name_scope("placeholders"):
            self.x = tf.placeholder(tf.float32, (None, 32, 32, 3))
            self.y = tf.placeholder(tf.int32, None)
            self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        with tf.name_scope("weights"):
            self.logits = LeNet(self.x, n_output_classes, self.keep_prob)
            self.predictions = tf.argmax(self.logits, 1)

        with tf.name_scope("loss"):
            one_hot_y = tf.one_hot(self.y, n_output_classes)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, one_hot_y)
            loss_operation = tf.reduce_mean(cross_entropy)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        self.training_operation = optimizer.minimize(loss_operation)
        correct_predictions = tf.equal(self.predictions, tf.argmax(one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def train(self, X_train, y_train, X_valid=None, y_valid=None, epochs=EPOCHS):
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        num_examples = len(X_train)
        max_validation_accuracy = 0

        print("Training...")
        print()
        for i in range(epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                self.sess.run(self.training_operation, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: .5})

            training_accuracy_dropout = self.sess.run(self.accuracy_operation,
                                                      {self.x: X_train, self.y: y_train, self.keep_prob: .5})
            training_accuracy = self.sess.run(self.accuracy_operation,
                                              {self.x: X_train, self.y: y_train, self.keep_prob: 1})
            validation_accuracy = self.sess.run(self.accuracy_operation,
                                                {self.x: X_valid, self.y: y_valid, self.keep_prob: 1})

            print("EPOCH {} ...".format(i + 1))
            print("Train Accuracy With Dropout= {:.3f}".format(training_accuracy_dropout))
            print("Train Accuracy = {:.3f}".format(training_accuracy))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))

            if validation_accuracy > max_validation_accuracy:
                max_validation_accuracy = validation_accuracy
                self.saver.save(self.sess, save_path())
                print("Model saved")
                print()

    def predict_prob(self, X):
        """ Return Probabilities of each prediction """
        return self.sess.run(self.logits, {self.x: X, self.keep_prob: 1})

    def predict(self, X):
        """ Return predicted class """
        return self.sess.run(self.predictions, {self.x: X, self.keep_prob: 1})

    def accuracy(self, X_test, y_test):
        """ Return model accuracy on a dataset """
        results = self.sess.run(self.accuracy_operation, {self.x: X_test, self.y: y_test, self.keep_prob: 1})
        print("Test Accuracy = {:.3f}".format(results))
        return results

    def restore(self, load_path: str):
        loader = tf.train.import_meta_graph(load_path)
        loader.restore(self.sess, tf.train.latest_checkpoint('./models'))


def normalize_multichannel(features):
    means = np.mean(features, axis=(0, 1, 2))
    centered = features - means
    stds = np.std(centered, axis=(0, 1, 2))
    return centered / stds


def save_path():
    models_path = os.path.join(".", MODELS_DIR)
    save_path = os.path.join(models_path, CURRENT_MODEL_NAME)
    return save_path


if __name__ == '__main__':
    main()
