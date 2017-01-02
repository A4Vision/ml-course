#!/usr/bin/python2.7
import sys
import os

import time

import collections

import itertools

SHOW = True
import matplotlib
if not SHOW:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
# Allow any further imports to happen smoothly.
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import sgd
import mnist_data
import tabulate


class LinearMulticlassSVM(sgd.ScoreCalculator):
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
        t = w - w[y_i]

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
    def __init__(self, w):
        self._w = w

    def predict(self, x):
        return np.argmax(np.dot(self._w, x.T), axis=0)

    def accuracy(self, x, y):
        return np.sum(self.predict(x) == y) / float(len(y))


def grid_search(accuracy_function, Cs, etas):
    accuracies = {}
    train_accuracies  =
    for C, eta in itertools.product(Cs, etas):
        train_accuracy, validation_accuracy = accuracy_function(C, eta)
        accuracies[()]


def find_C_eta():
    train_labels = np.array(mnist_data.train_labels, dtype=np.int32)
    validation_labels = np.array(mnist_data.validation_labels, dtype=np.int32)
    validation_errors = collections.defaultdict(list)
    train_errors = collections.defaultdict(list)
    descents = {}
    etas = [10 ** -i for i in xrange(1, 11)]
    Cs = [10 ** i for i in xrange(-4, 2)]
    for eta in etas:
        for C in Cs:
            m, d = mnist_data.train_data.shape
            k = 10
            w0 = np.zeros((k, d))
            score = LinearMulticlassSVM(C)
            descents[(eta, C)] = sgd.StochasticGradientDescent(w0, eta, score)

    for i in xrange(5):
        print 'i=', i
        for eta in etas:
            for C in Cs:
                descent = descents[(eta, C)]
                descent.run(5000, mnist_data.train_data, train_labels)
                predictor = LinearMulticlassSVMPredictor(descent.w())
                accuracy = predictor.accuracy(mnist_data.validation_data, validation_labels)
                validation_errors[(eta, C)].append(1. - accuracy)
                train_accuracy = predictor.accuracy(mnist_data.train_data, train_labels)
                train_errors[(eta, C)].append(1. - train_accuracy)
        print tabulate.tabulate(dict2table(validation_errors, 'eta', 'C'))


def dict2table(d, x_name, y_name):
    x_values = sorted(set(key[0] for key in d))
    y_values = sorted(set(key[1] for key in d))
    table = [[None] * len(x_values) for _ in xrange(len(y_values))]
    for (x, y), value in d.iteritems():
        table[y_values.index(y)][x_values.index(x)] = value
    first_row = [None] + ['{}={}'.format(x_name, x) for x in x_values]
    table.insert(0, first_row)
    for i, y in enumerate(y_values):
        table[1 + i].insert(0, '{}={}'.format(y_name, y))
    return table


def numpy_stuff():
    a = np.array([[1, 2, 3,], [4, 5, 6.]])
    print 'distance.norm(a)', np.sum(a ** 2, axis=1)
    print 'a[np.array([0, 1, 0, 1, 1, 1])]', a[np.array([0, 1, 0, 1, 1, 1])]
    print 'a', a
    print 'a - a[1]', a - a[1]
    print 'np.dot(a, np.array([3, 1, 0.]))', np.dot(a, np.array([3, 1, 0.]))
    a[0] += np.array([1., 2., 0.])
    print 'a', a


def main(output_directory):
    x = time.time()
    find_C_eta()
    print 'total', time.time() - x

if __name__ == '__main__':
    main("")
