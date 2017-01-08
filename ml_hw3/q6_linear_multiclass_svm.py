#!/usr/bin/python2.7
import sys
import os
import time
import collections
import itertools

import functools


SHOW = False
import matplotlib
if not SHOW:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

# Allow any further imports to happen smoothly.
sys.path.insert(0, os.path.dirname(__file__))
import sgd
import mnist_data
import parameters_grid_search


class LinearMulticlassSVM(sgd.ScoreCalculator):
    """
    Calculates the SVM score, and a sub-gradient of it.
    """

    def __init__(self, C):
        self._C = C

    def gradient(self, w, x_i, y_i):
        """
        :param w:
            w is an kXd matrix:
                w_j = w[j]
        :param x_i: a row of length d
        :param y_i: a scalar
        :return: gradient of:
            0.5 * ||w|| ** 2 + max(0, max_j((w[j] - w[y_i]) * x_i + 1; for all j != y_i))
        """
        gradient = w.copy()
        loss_values = np.dot(w - w[y_i], x_i) + 1
        loss_values[y_i] = 0
        j_0 = np.argmax(loss_values)
        l_i = loss_values[j_0]
        if l_i > 0:
            gradient[y_i] -= x_i * self._C
            gradient[j_0] += x_i * self._C
        return gradient

    def score(self, w, x, y):
        """
        :param w:
            w is a kXd matrix:
                w[j] = w_j
        :param x: The training inputs.
            x is an mXd matrix:
                x[i] = x_i
        :param y: The labels.
            vector of length m.
        :return:
        """
        m, d = x.shape
        k, d1 = w.shape
        m1 = y.shape[0]
        assert d1 == d
        assert m1 == m
        loss = 0
        for i in xrange(m):
            loss_values = np.dot(w - w[y[i]], x[i]) + 1
            loss_values[y[i]] = 0
            assert loss_values.shape == (k,)
            loss += np.max(loss_values)
        return np.sum(0.5 * np.sum(w ** 2, axis=1)) + loss * self._C / m


class LinearMulticlassSVMPredictor(object):
    """
    Predicts a label for SVM using a matrix w:
        label(x) = argmax_j(w_j * x)
    """

    def __init__(self, w):
        """
        :param w: numpy array of shape (k, d).
            Coefficients matrix of size kXd.
            k: number of possible labels
            d: dimension of the data.
            w[j] = w_j, the j'th vector that is used to predict the j'th label.
        """
        self._w = w

    def predict(self, x):
        """
        :param x: numpy array of size mXd.
            each row, x[i], is a data point to label.
        :return: numpy array of length m.
            ret_val[i] = argmax_j(w_j * x_i)
        """
        return np.argmax(np.dot(self._w, x.T), axis=0)

    def k(self):
        """
        Amount of labels.
        :return:
        """
        return self._w.shape[1]

    def accuracy(self, x, y):
        """
        :param x: numpy array of shape (m, d).
            each row, x[i], is a data point to label.
        :param y: numpy array of length m.
         m[i]: scalar, the actual label of x[i].
                number between 1 and k
        :return: (Number of correct predictions for the given samples) / (amount of samples)
        """
        return np.sum(self.predict(x) == y) / float(len(y))


def measure_accuracies_wrapper(N):
    permutation = np.random.permutation(mnist_data.train_data.shape[0])
    train_labels = np.array(mnist_data.train_labels, dtype=np.int32)[permutation][:N].copy()
    train_data = np.array(mnist_data.train_data, dtype=np.float32)[permutation][:N].copy()
    return functools.partial(measure_accuracies, train_labels=train_labels[:N], train_data=train_data[:N])


def measure_accuracies(C, eta, iterations, train_data, train_labels):
    """
    Measure the training accuracy, and the validation accuracy of the multiclass SVM
    classifier.

    :param train_data:
    :param train_labels:
    :param C: punishment constant for the score function.
    :param eta: step size for SGD.
    :param iterations: amount of iterations for SGD.
    :return: AccuracyMeasurement
    """
    validation_labels = np.array(mnist_data.validation_labels, dtype=np.int32)
    m, d = train_data.shape
    # Number of labels.
    k = 10
    w0 = np.zeros((k, d))
    score = LinearMulticlassSVM(C)
    descent = sgd.StochasticGradientDescent(w0, eta, score)
    descent.run(iterations, train_data, train_labels)
    predictor = LinearMulticlassSVMPredictor(descent.w())
    accuracy_validation = predictor.accuracy(mnist_data.validation_data, validation_labels)
    accuracy_train = predictor.accuracy(train_data, train_labels)

    return parameters_grid_search.AccuracyMeasurement(validation=accuracy_validation, train=accuracy_train)


def test_set_accuracy(C, eta, iterations, output_directory):
    test_data = np.array(mnist_data.test_data, dtype=np.float32)
    train_data = np.array(mnist_data.train_data, dtype=np.float32)
    test_labels = np.array(mnist_data.test_labels, dtype=np.int32)
    train_labels = np.array(mnist_data.train_labels, dtype=np.int32)
    m, d = train_data.shape
    # Number of labels.
    k = 10
    w0 = np.zeros((k, d))
    score = LinearMulticlassSVM(C)
    descent = sgd.StochasticGradientDescent(w0, eta, score)
    descent.run(iterations, train_data, train_labels)
    descent.w().dump(open(os.path.join(output_directory, "tmp.dat"), "wb"))
    for j, w_j in enumerate(descent.w()):
        plt.imsave(os.path.join(output_directory, "w_{}.png".format(j)), np.reshape(w_j, (28, 28)))

    predictor = LinearMulticlassSVMPredictor(descent.w())
    return predictor.accuracy(test_data, test_labels)


def main(output_directory):
    np.array([]).dump(open(os.path.join(output_directory, "tmp.dat"), "wb"))
    np.random.seed(123)
    np.seterr(all='ignore')
    x = time.time()
    # Apply a naive grid search, using only 10000 iterations, and 10000 data points.
    best_C, best_eta = parameters_grid_search.best_C_eta(measure_accuracies_wrapper,
                                                         gross_search_iterations=20000,
                                                         n_gross_search_samples=10000)
    # Plot accuracy around the best eta and best C found, using all the data, and 50000 iterations.
    parameters_grid_search.generate_error_rate_plots(best_C, best_eta,
                                                     measure_accuracies_wrapper(N=50000), 50000,
                                                     output_directory, "q6_Linear_SVM")
    print 'best: C={}, eta={}'.format(best_C, best_eta)
    # Calculate the accuracy on the test set, using all the data, 200000 iterations.
    print 'test set accuracy:', test_set_accuracy(best_C, best_eta, 200000, output_directory)
    print 'Total runtime', time.time() - x


if __name__ == '__main__':
    main(os.path.dirname(__file__))
