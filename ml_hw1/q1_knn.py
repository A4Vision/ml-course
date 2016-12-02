import math
import collections
import itertools
import os

import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from sklearn.datasets import fetch_mldata
import numpy
import numpy.linalg


def _InitGlobals(silly_random):
    """Initialize the training and test data."""
    global mnist, data, labels, train, train_labels, test, test_labels
    print "fetching MNIST data...",
    mnist = fetch_mldata("MNIST original")
    print "done"
    data = mnist["data"]
    labels = mnist["target"]
    if silly_random:
      idx = numpy.random.RandomState(0).choice(70000, 11000)
    else:
      # randomize without repetitions
      import random
      unique_idx = random.sample(range(70000), 11000)
      idx = numpy.array(unique_idx)
    train = data[idx[:10000], :].astype(numpy.float32)

    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(numpy.float32)
    test_labels = labels[idx[10000:]]


def Accuracy(predictions, labels):
    assert numpy.array(predictions).shape == labels.shape
    correct = (numpy.array(predictions) == labels)
    return correct.sum() / float(len(correct))


def MostCommon(labels_vec):
    return collections.Counter(labels_vec).most_common(1)[0][0]


# Q1_a
def kNN(images, labels, query_image, k):
    assert len(images) == len(labels)
    assert k <= len(images)
    assert len(images) > 0
    assert images[0].shape == query_image.shape

    distances = numpy.linalg.norm(images - query_image, axis=1)
    best_k_indices = numpy.argpartition(distances, k)[:k]

    k_labels = labels[best_k_indices]
    return MostCommon(k_labels)


def BinomialAccuracyStandardDeviation(n, p):
    q = 1. - p
    return math.sqrt(n * p * q) / n


def Q1_b(n):
    question_train_labels = train_labels[:n]
    question_train = train[:n]
    question_test = test

    print "::Question 1b::"
    k = 10
    predictions = [kNN(question_train, question_train_labels, query_sample, k) for query_sample in question_test]
    accuracy = Accuracy(predictions, test_labels)
    print "\tAccuracy: {:.5f}".format(accuracy)

    n_samples = len(test)
    p = 1. / 10
    random_predictor_std = BinomialAccuracyStandardDeviation(n_samples, p)
    print "\tRandomPredictor: {:.5f} +- std= {:.5f}".format(p, random_predictor_std)


class OptimizedKNN(object):
    """
    Faster K-Nearest Neighbors, assuming that one
    needs to use the same training set and test set
    with varrying k values, or using only smaller parts of the training set.
    Usage:
      >>> test, test_labels = load_test()
      >>> knn = OptimizedKnn(train, train_labels, test)
        Many times with the same index but usign various k and n:
          >>> knn.LabelByNearest(test_index, k, n) == test_labels[test_index]

    """
    def __init__(self, all_train, all_train_labels, all_test):
        # One-Time preprocess.
        self._sorted_train_indices = self._calculate_distances(all_train, all_test)
        self._all_train_labels = all_train_labels

    def _calculate_distances(self, all_train, all_test):
        """
        Calculates distances for every pair:
          (training sample, test sample)
        Returns a mapping:
            test-sample-index --> sorted-indices
          sorted-indices is a sorted is the list of indices of all training samples
          sorted according to distance from all_test[test-sample-index].
        """
        sorted_indices = []
        for sample in all_test:
            distances = numpy.linalg.norm(all_train - sample, axis=1)
            indices = numpy.argsort(distances)
            sorted_indices.append(indices)
        return sorted_indices

    def LabelByNearest(self, test_sample_index, k, n):
        """
        Label the specified test samle, using k-NN with given k,
        using only the first n samples of the training set.

        :param test_sample_index: what test sample to label
        :param k: The k for k-NN algorithm.
        :param n: How many training samples to use.
        """
        assert 0 < n <= len(self._all_train_labels)
        # Indices of all train examples, sorted by distance.
        indices = self._sorted_train_indices[test_sample_index]

        # Take only indices under n, assuming we train using only the first n samples.
        relevant_indices = indices[indices < n]
        if n == len(self._all_train_labels):
            assert (relevant_indices == indices).all()
        # Indices of k nearest train samples.
        k_nearest_indices = relevant_indices[:k]
        k_labels = self._all_train_labels[k_nearest_indices]
        return MostCommon(k_labels)


def Q1_c(n, output_directory, show=False):
    question_train_labels = train_labels[:n]
    question_train = train[:n]
    question_test = test

    print "::Question 1c::"
    pre_start = time.time()
    knn = OptimizedKNN(question_train, question_train_labels, question_test)
    print "\tpreprocess runtime:", time.time() - pre_start
    accuracies = []
    k_values = range(1, 101)
    eval_start = time.time()
    for k in k_values:
        predictions = [knn.LabelByNearest(test_sample_index, k, n) for test_sample_index, _ in enumerate(question_test)]
        accuracies.append(Accuracy(predictions, test_labels))
    print "\tevaluate runtime:", time.time() - eval_start
    pyplot.plot(k_values, accuracies, 'bo--')
    pyplot.xlabel("k (of k NearestNeighbors)")
    pyplot.ylabel("MNIST prediction accuracy")
    pyplot.title("Question 1c:: k-Nearest Neighbors accuracy as a function of k.")
    output_file = os.path.join(output_directory, "Q1c.png")
    pyplot.savefig(output_file)
    if show:
      os.system(output_file)
    pyplot.clf()


def Q1_d(n_values, output_directory):
    max_n = max(n_values)
    question_train_labels = train_labels[:max_n]
    question_train = train[:max_n]
    question_test = test
    best_k = 1

    print "::Question 1d::"
    preprocess_start = time.time()
    # NOTE: For code simplicity, we don't randomize m samples every time, but use overlapping samples.
    # The first 100, then the first 200, ... then the first 5000.
    knn = OptimizedKNN(question_train, question_train_labels, question_test)
    print "\tpreprocess runtime:", time.time() - preprocess_start
    accuracies = []
    eval_start = time.time()
    for n in n_values:
        predictions = [knn.LabelByNearest(test_sample_index, best_k, n) for test_sample_index, _ in enumerate(question_test)]
        accuracies.append(Accuracy(predictions, test_labels))
    print "\tevaluate runtime:", time.time() - eval_start
    pyplot.plot(n_values, accuracies, 'ro--')
    pyplot.title("Question 1d :: kNN accuracy as a function of train samples, using k={}".format(best_k))
    pyplot.xlabel("n (number of training sample")
    pyplot.ylabel("MNIST prediction accuracy")
    pyplot.savefig(os.path.join(output_directory, "Q1d.png"))
    pyplot.clf()


def main(output_directory, silly_random=True):
    numpy.random.seed(123)
    _InitGlobals(silly_random)
    start_b = time.time()
    Q1_b(n=1000)
    print "\t1b runtime:", time.time() - start_b
    start_c = time.time()
    Q1_c(n=1000, output_directory=output_directory)
    print "\t1c runtime:", time.time() - start_c
    start_d = time.time()
    Q1_d(n_values=range(100, 5001, 100), output_directory=output_directory)
    print "\t1d runtime:", time.time() - start_d
