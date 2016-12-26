import random
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
SHOW = False
import tabulate
import matplotlib
if not SHOW:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
import mnist_data
N_runs = 100



class Perceptron(object):
    """
    Naive implementation of the Preceptron algorithm.
    Find a vector w, such that sign(<w, x>) ~= y
    """

    def __init__(self, w_0):
        assert w_0.ndim == 1
        self._w = w_0

    def predict(self, x):
        """
        sign(wT * x)
        :param x:
        :return:
        """
        return np.sign(np.inner(self._w, x))

    def update(self, x_i, y_i):
        assert y_i in (-1., 1.)
        if self.predict(x_i) != y_i:
            self._w += y_i * x_i

    def get_w(self):
        return np.array(self._w)

    @staticmethod
    def train_on_data(data, data_labels):
        perceptron = Perceptron(np.zeros_like(data[0]))
        for x_i, y_i in zip(data, data_labels):
            perceptron.update(x_i, y_i)
        return perceptron

    def accuracy(self, data, labels):
        return np.sum(np.sign(np.dot(data, self._w)) == labels) / float(len(data))


def get_percentile(scores, percentile):
    assert 0 <= percentile <= 1.
    n = len(scores)
    position = min(int(round(n * percentile)), n - 1)
    return sorted(scores)[position]


def get_accuracies(train_data, train_labels, test_data, test_labels, n_runs):
    """
    Run the Perceptron on the training data with various random permutations, retrieve
    the accuracies.

    :param train_data:
    :param train_labels:
    :param test_data:
    :param test_labels:
    :param n_runs:
    :return:
    """
    accuracies = []
    for i in xrange(n_runs):
        order = np.random.permutation(len(train_data))
        ordered_train_data = train_data[order]
        ordered_train_labels = train_labels[order]
        perceptron = Perceptron.train_on_data(ordered_train_data, ordered_train_labels)
        accuracy = perceptron.accuracy(test_data, test_labels)
        accuracies.append(accuracy)
    return accuracies


def transpose_table(table):
    """
    ret_val[i][j] == table[j][i]
    :param table:
    :return:
    """
    return map(list, zip(*table))


def q1_a(output_directory, normalized_train_data, normalized_test_data):
    print "Q1a"
    percentile_95 = []
    percentile_5 = []
    average_accuracy = []
    n_values = (5, 10, 100, 500, 1000, 5000)
    for n in n_values:
        samples = normalized_train_data[:n]
        samples_labels = mnist_data.train_labels[:n]
        accuracies = get_accuracies(samples, samples_labels, normalized_test_data, mnist_data.test_labels, N_runs)
        percentile_95.append(get_percentile(accuracies, 0.95))
        percentile_5.append(get_percentile(accuracies, 0.05))
        average_accuracy.append(np.average(accuracies))
    print tabulate.tabulate(transpose_table([n_values, average_accuracy, percentile_5, percentile_95]),
                            headers=("n (training data)", "Average", "Percentile %5", "Percentile %95"))
    plt.plot(n_values, average_accuracy, "red", marker="*", label="Average accuracy")
    plt.plot(n_values, percentile_95, "blue", marker="+", label="Precentile %95 accuracy")
    plt.plot(n_values, percentile_5, "green", marker="o", label="Precentile %5 accuracy")
    plt.legend(loc=4)
    plt.savefig(os.path.join(output_directory, "Q1a_PreceptronAccuracies.png"))
    if SHOW:
        plt.show()
    plt.cla()


def q1_b(output_directory, normalized_train_data, normalized_test_data):
    perceptron = Perceptron.train_on_data(normalized_train_data, mnist_data.train_labels)
    image = perceptron.get_w()
    outfile = os.path.join(output_directory, "Q1b_PreceptronWeights.png")
    plt.imsave(outfile, np.reshape(image, (28, 28)))
    if SHOW:
        plt.imshow(np.reshape(image, (28, 28)), interpolation='nearest')
        plt.show()
    plt.cla()


def q1_c(normalized_train_data, normalized_test_data):
    print "Q1c"
    perceptron = Perceptron.train_on_data(normalized_train_data, mnist_data.train_labels)
    accuracy = perceptron.accuracy(normalized_test_data, mnist_data.test_labels)
    print "Accuracy over all data =", accuracy


def q1_d(output_directory, normalized_train_data, normalized_test_data):
    print "Q1d"
    perceptron = Perceptron.train_on_data(normalized_train_data, mnist_data.train_labels)
    missclassified = [i for i, (sample, label) in enumerate(zip(normalized_test_data, mnist_data.test_labels))
                      if perceptron.predict(sample) != label]
    random.seed(123)
    i = random.choice(missclassified)
    original_sample = mnist_data.test_data_unscaled[i]
    outfile = os.path.join(output_directory, "Q1d_PreceptronMissclassified.png")
    plt.imsave(outfile, np.reshape(original_sample, (28, 28)))
    if SHOW:
        plt.matshow(np.reshape(original_sample, (28, 28)))
        plt.show()
    plt.cla()


def main(dir_name):
    # Normalize the data - ||x|| == 1
    normalized_train_data = preprocessing.normalize(mnist_data.train_data, axis=1)
    normalized_test_data = preprocessing.normalize(mnist_data.test_data, axis=1)
    q1_a(dir_name, normalized_train_data, normalized_test_data)
    q1_b(dir_name, normalized_train_data, normalized_test_data)
    q1_c(normalized_train_data, normalized_test_data)
    q1_d(dir_name, normalized_train_data, normalized_test_data)

if __name__ == '__main__':
    dir_name = os.path.join(os.path.dirname(__file__), "outputs")
    main(dir_name)


