#!/usr/bin/python2.7
import sys
import os
import time
import functools
import cProfile
import datetime

# Amount kernels we calculate and keep in memory,
# during optimization.
BATCH_SIZE = 25000
# Amount kernels we calculate and keep in memory,
# during prediction.
SMALL_BATCH_SIZE = 1000
MEMORY_LIMIT_MB = 1024 * 3

import numpy as np

# Allow any further imports to happen smoothly.
sys.path.insert(0, os.path.dirname(__file__))
import mnist_data
import parameters_grid_search


class KernelCalculator(object):
    """
    Implements the actual Kernel function calculation.
    Calculates K(x_i, y_i).
    """

    def kernel_matrix(self, x1, x2):
        raise NotImplementedError


class LinearKernel(KernelCalculator):
    def kernel_matrix(self, x1, x2):
        print 'Allocating for kernels: {:.0f}MB'.format(x1.shape[0] * x2.shape[0] * 4 / 2 ** 20.)
        assert x1.shape[0] * x2.shape[0] * 4 / 2 ** 20. < MEMORY_LIMIT_MB
        return np.dot(x1, x2.T)


class QuadraticKernel(KernelCalculator):
    def __init__(self, c):
        self._c = c

    def kernel_matrix(self, x1, x2):
        print 'Allocating for kernels: {:.0f}MB'.format(x1.shape[0] * x2.shape[0] * 4 / 2 ** 20.)
        assert x1.shape[0] * x2.shape[0] * 4 / 2 ** 20. < MEMORY_LIMIT_MB
        return (np.dot(x1, x2.T) + self._c) ** 2


class BatchKernel(object):
    """
    Calculates kernels in batches.
    """

    def __init__(self, data, kernel_calculator):
        self._data = data
        self._min_i = None
        self._max_i = None
        self._preprocessed_kernels = None
        assert isinstance(kernel_calculator, KernelCalculator)
        self._kernel_calculator = kernel_calculator

    def kernel_with_unknown(self, x):
        """
        :param x:
        :return: np.array([K(x, data[j]) for j in xrange(len(data))])
        """
        return self._kernel_calculator.kernel_matrix(self._data, x)

    def kernels_with_known(self, i):
        """
        The i'th row of the kernel matrix.

        :param i:
        :return: np.array([K(data[i], data[j]) for j in xrange(len(data))])
        """
        assert self._min_i <= i <= self._max_i
        return self._preprocessed_kernels[i - self._min_i]

    def preprocess_full(self):
        self.preprocess(0, self.data_length() - 1)

    def preprocess(self, min_i, max_i):
        """
        Do some preprocess, that enables calculations of
            kernels_with_known(i) for i between min_i and max_i, including both ends.
        :param min_i:
        :param max_i:
        """
        if self._min_i is not None and self._min_i <= min_i and max_i <= self._max_i:
            return
        assert 0 <= min_i <= max_i <= len(self._data) - 1
        self._min_i = min_i
        self._max_i = max_i
        del self._preprocessed_kernels
        self._preprocessed_kernels = self._kernel_calculator.kernel_matrix(self._data[min_i: max_i + 1], self._data)

    def data_length(self):
        """
        Amount of data rows.
        :return: m, amount of data rows.
        """
        return len(self._data)


class KernelGradientCalculator(object):
    """
    The score function -
        Calculates the score,
        Calculates the gradient/stochastic gradient.
    """

    def __init__(self, C):
        self._C = C

    def gradient(self, v, i, y_i, K_i):
        gradient = v.copy()
        loss_values = np.dot(v - v[y_i], K_i) + 1
        loss_values[y_i] = 0
        j_0 = np.argmax(loss_values)
        l_i = loss_values[j_0]
        if l_i > 0:
            gradient[y_i][i] -= self._C
            gradient[j_0][i] += self._C
        return gradient


class StochasticGradientDescent(object):
    """
    Implements (stochastic) gradient descent:
        x -= eta * gradient(x)
    """

    def __init__(self, v0, eta, gradient_calculator, batch_kernel):
        assert isinstance(gradient_calculator, KernelGradientCalculator)
        self._gradient_calculator = gradient_calculator
        assert isinstance(batch_kernel, BatchKernel)
        self._batch_kernel = batch_kernel
        self._v = v0
        self._eta = eta

    def step(self, i, y_i):
        """

        :param i:
        :param y_i:
        :return:
        """
        kernel = self._batch_kernel.kernels_with_known(i)
        gradient = self._gradient_calculator.gradient(self._v, i, y_i, kernel)
        self._v -= self._eta * gradient

    def v(self):
        return self._v.copy()

    def run(self, n_steps, y):
        for i in xrange(n_steps):
            i_actual = i % self._batch_kernel.data_length()
            if i_actual % BATCH_SIZE == 0:
                start = i_actual
                end = min(start + BATCH_SIZE - 1, self._batch_kernel.data_length() - 1)
                self._batch_kernel.preprocess(start, end)
            self.step(i_actual, y[i_actual])


class KernelMulticlassSVMPredictor(object):
    """
    Predicts a label for SVM using a matrix w:
        label(x) = argmax_j(w_j * x)
    """

    def __init__(self, v, batch_kernel_calculator):
        """
        :param w: numpy array of shape (k, d).
            Coefficients matrix of size kXd.
            k: number of possible labels
            d: dimension of the data.
            w[j] = w_j, the j'th vector that is used to predict the j'th label.
        """
        self._v = v
        assert isinstance(batch_kernel_calculator, BatchKernel)
        self._batch_kernel_calculator = batch_kernel_calculator

    def predict(self, x):
        """
        :param x: numpy array of size mXd.
            each row, x[i], is a data point to label.
        :return: numpy array of length m.
            ret_val[i] = argmax_j(w_j * x_i)
        """
        kernels = None
        res = np.array([], dtype=np.int32)
        for i in xrange(0, len(x), SMALL_BATCH_SIZE):
            del kernels
            kernels = self._batch_kernel_calculator.kernel_with_unknown(x[i:i + SMALL_BATCH_SIZE])
            res = np.concatenate((res, np.argmax(np.dot(self._v, kernels), axis=0)))
        return res

    def k(self):
        """
        Amount of labels.
        :return:
        """
        return self._v.shape[1]

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
    kernel_calculator = QuadraticKernel(1)
    batch_kernel = BatchKernel(train_data, kernel_calculator)
    return functools.partial(measure_accuracies, batch_kernel=batch_kernel,
                             train_data=train_data, train_labels=train_labels)


def measure_accuracies(C, eta, iterations, batch_kernel, train_data, train_labels):
    """
    Measure the training accuracy, and the validation accuracy of the multiclass SVM
    classifier.

    :type train_labels: np.arary
    :param train_data:
    :param batch_kernel:
    :param C: punishment constant for the score function.
    :param eta: step size for SGD.
    :param iterations: amount of iterations for SGD.
    :return: AccuracyMeasurement
    """
    validation_labels = np.array(mnist_data.validation_labels, dtype=np.int32)
    validation_data = np.array(mnist_data.validation_data, dtype=np.float32)
    m, d = train_data.shape
    # Number of labels.
    k = 10

    k = 10
    v0 = np.zeros((k, m))

    gradient_calculator = KernelGradientCalculator(C)

    print 'Descending...'
    descent = StochasticGradientDescent(v0, eta, gradient_calculator, batch_kernel)
    descent.run(iterations, train_labels)
    print 'Calculating accuracy...'
    predictor = KernelMulticlassSVMPredictor(descent.v(), batch_kernel)
    accuracy_validation = predictor.accuracy(validation_data, validation_labels)
    accuracy_train = predictor.accuracy(train_data, train_labels)

    return parameters_grid_search.AccuracyMeasurement(validation=accuracy_validation, train=accuracy_train)


def main(output_directory):
    np.random.seed(123)
    p = cProfile.Profile()
    p.enable()
    for x in ('MKL_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[x] = '3'
    x = time.time()
    accuracies_gross, accuracies_deep = parameters_grid_search.best_C_eta(measure_accuracies_wrapper,
                                                   gross_search_iterations=2000, n_gross_search_samples=2000,
                                                   deep_search_iterations=2000, n_deep_search_samples=2000)

    print 'Total runtime', time.time() - x
    p.disable()
    p.dump_stats("q7{}.prof".format(datetime.datetime.now().strftime("%d-%m %H-%M-%S")))


if __name__ == '__main__':
    main("")
