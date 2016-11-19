import math
import collections
import itertools
import os

import time
from matplotlib import pyplot
from sklearn.datasets import fetch_mldata
import numpy
import numpy.linalg


def _InitGlobals():
    """Initialize the training and test data."""
    global mnist, data, labels, train, train_labels, test, test_labels
    print "fetching MNIST data...",
    mnist = fetch_mldata("MNIST original")
    print "done"
    data = mnist["data"]
    labels = mnist["target"]
    idx = numpy.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :]

    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :]
    test_labels = labels[idx[10000:]]


def Accuracy(predictions, labels):
    assert numpy.array(predictions).shape == labels.shape
    correct = (numpy.array(predictions) == labels)
    return correct.sum() / float(len(correct))


def MostCommon(labels_vec):
    return collections.Counter(labels_vec).most_common(1)[0][0]


def NormalizeGaussian(train, test):
    mean = numpy.average(train, axis=0)
    std = numpy.std(train, axis=0)
    std[std == 0] = 1.
    return (train - mean) / std, (test - mean) / std


def NormalizeUniform(train, test):
    _min = numpy.min(train, axis=0)
    _max = numpy.max(train, axis=0)
    diff = _max - _min
    diff[diff == 0] = 1.
    return (train - _min) / diff, (test - _min) / diff


def kNN(images, labels, query_image, k):
    # Q1_a
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
    question_train, question_test = NormalizeGaussian(train[:n], test)

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
    def __init__(self, all_train, all_train_labels, all_test):
        self._sorted_train_indices = self._calculate_distances(all_train, all_test)
        self._all_train_labels = all_train_labels

    def _calculate_distances(self, all_train, all_test):
        sorted_indices = []
        for sample in all_test:
            distances = numpy.linalg.norm(all_train - sample, axis=1)
            indices = numpy.argsort(distances)
            sorted_indices.append(indices)
        return sorted_indices

    def LabelByNearest(self, test_sample_index, k, n):
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


def Q1_c(n, output_directory):
    question_train_labels = train_labels[:n]
    question_train, question_test = NormalizeGaussian(train[:n], test)

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
    pyplot.savefig(os.path.join(output_directory, "Q1c.png"))
    pyplot.clf()


def Q1_d(n_values, output_directory):
    max_n = max(n_values)
    question_train_labels = train_labels[:max_n]
    # Normalize test and training samples according to first 100 train samples.
    # This results in worse results - but it simplifies the code.
    _, question_test = NormalizeGaussian(train[:min(n_values)], test)
    _, question_train = NormalizeGaussian(train[:min(n_values)], train[:max_n])
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
    pyplot.xlabel("n (number of training samples - NOTE: normalized only by first 100)")
    pyplot.ylabel("MNIST prediction accuracy")
    pyplot.savefig(os.path.join(output_directory, "Q1d.png"))
    pyplot.clf()


def main(output_directory):
    numpy.random.seed(123)
    _InitGlobals()
    start_b = time.time()
    Q1_b(n=1000)
    print "\t1b runtime:", time.time() - start_b
    start_c = time.time()
    Q1_c(n=1000, output_directory=output_directory)
    print "\t1c runtime:", time.time() - start_c
    start_d = time.time()
    Q1_d(n_values=range(100, 5001, 100), output_directory=output_directory)
    print "\t1d runtime:", time.time() - start_d
