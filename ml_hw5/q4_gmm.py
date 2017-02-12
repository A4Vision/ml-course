import collections
import os
import time

from scipy.spatial import distance

SHOW = True
import matplotlib
if not SHOW:
    matplotlib.use('Agg')

from matplotlib import pyplot as plt
import math
from ml_hw5 import mnist_data
import numpy as np


class GaussianMixtureModelOptimizer(object):
    def __init__(self, gmm):
        self._gmm = gmm

    def _latent_probs(self, data_len):
        likelihoods = [self._gmm.weighted_likelihoods(i)
                       for i in xrange(data_len)]
        return [weighted / np.sum(weighted) for weighted in likelihoods]

    def step(self, data):
        self._gmm.preprocess(data)
        # latent_probs[i][m] = Prob(z_i = m | x_i, theta_t)
        latent_probs = np.array(self._latent_probs(len(data)))
        for m in xrange(len(self._gmm)):
            latent_sum = np.sum(latent_probs.T[m])
            u_m = 1. / latent_sum * np.dot(data.T, latent_probs.T[m])
            # plt.imshow(u_m.reshape(28, 28))
            # plt.show()
            distances = self._gmm.distances(m)
            sigma_m_squared = 1. / latent_sum * np.inner(distances, latent_probs.T[m])
            n = len(data)
            c_m = 1. / n * latent_sum
            self._gmm.update_gaussian(m, c_m, Gaussian(u_m, sigma_m_squared ** 0.5))


class GaussianMixtureModel(object):
    def __init__(self, c_aprior_probs, gaussians):
        assert len(gaussians) == len(c_aprior_probs)
        self._c = np.array(c_aprior_probs, dtype=np.float)
        self._gaussians = list(gaussians)

    def __len__(self):
        return len(self._c)

    def preprocess(self, data):
        for gaussian in self._gaussians:
            gaussian.prepare_distances(data)

    def likelihood(self, i):
        res = 0.
        for c, gaussian in zip(self._c, self._gaussians):
            res += c * gaussian.likelihood(i)
        return res

    def weighted_likelihoods(self, i):
        return np.array(self._c) * np.array([gaussian.likelihood(i) for gaussian in self._gaussians])

    def distances(self, m):
        return self._gaussians[m].distances()

    def update_gaussian(self, m, c, gaussian):
        self._c[m] = c
        self._gaussians[m] = gaussian


class Gaussian(object):
    def __init__(self, u, sigma):
        self._sigma = sigma
        self._u = u
        self._distances = np.array([], dtype=np.float)

    def prepare_distances(self, data):
        self._distances = np.sum((data - self._u) ** 2, axis=1)
        assert abs(self._distances[1] - distance.norm(data[1] - self._u, ord=2) ** 2) < 0.00001
        assert len(self._distances) == len(data)

    def likelihood(self, i):
        assert 0 <= i < len(self._distances)
        return 1. / self._sigma * np.exp(-0.5 / self._sigma ** 2. * self._distances[i])

    def distances(self):
        return self._distances


def q4():
    mnist_data.train_data /= 256.
    mnist_data.test_data /= 256.
    labels = sorted(set(map(int, mnist_data.train_labels)))
    print labels
    n = len(labels)
    label2x = {}
    for label, x in zip(mnist_data.train_labels, mnist_data.train_data):
        label2x[label] = x
        if len(label2x) == n:
            break
    print collections.Counter(list(mnist_data.train_labels))
    gmm = GaussianMixtureModel([1. / n] * n, [Gaussian(label2x[label], 2.) for label in labels])

    optimizer = GaussianMixtureModelOptimizer(gmm)
    for j in xrange(20):
        print 'step', j
        optimizer.step(mnist_data.train_data)
        # if j == 1:
        #     for i in xrange(len(gmm)):
        #         plt.imshow(np.reshape(gmm._gaussians[i]._u , (28, 28)))
        #         plt.show()
        gmm.preprocess(mnist_data.train_data)
        print 'likelihood', sum([gmm.likelihood(i) for i in xrange(len(mnist_data.train_data))])
    for t in [0, 1]:
        plt.imshow(np.reshape(gmm._gaussians[t]._u , (28, 28)))
        plt.show()
    plt.cla()

    return gmm, [], []


def plot_likelihood_scores(output_directory, likelihood_scores):
    pass


def plot_final_state(output_directory, final_state):
    pass


def plot_accuracy_rates(output_directory, accuracy_rates):
    pass


def main(output_directory):
    print np.exp(1.)
    final_state, test_accuracy_rates, likelihood_scores = q4()
    plot_likelihood_scores(output_directory, likelihood_scores)
    plot_final_state(output_directory, final_state)
    plot_accuracy_rates(output_directory, test_accuracy_rates)


if __name__ == '__main__':
    main("outputs")