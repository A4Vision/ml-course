import os

from scipy.spatial import distance
import scipy.misc

SHOW = False
import matplotlib

if not SHOW:
    matplotlib.use('Agg')

from matplotlib import pyplot as plt
import mnist_data
import numpy as np


class GaussianMixtureModelOptimizer(object):
    """
    Finds the best GMM to fit the data.
    """

    def __init__(self, gmm):
        """
        :param gmm: starting point.
        :type gmm: GaussianMixtureModel
        """
        self._gmm = gmm

    def _latent_probs(self, data_len):
        """
        :param data_len:
        :return: list of arrays, latent_probs, s.t.:
            latent_probs[i][m] = Prob(z_i = m | x_i, theta_t)
        """
        log_likelihoods = [self._gmm.weighted_log_likelihoods(i)
                           for i in xrange(data_len)]
        log_probs = [weighted - scipy.misc.logsumexp(weighted) for weighted in log_likelihoods]
        return [np.exp(vec) for vec in log_probs]

    def step(self, data):
        """
        Optimize the current GMM to fit the given data.

        Runs in O(data_size) operations.
        :param data:
        :return:
        """
        self._gmm.preprocess(data)
        # latent_probs[i][m] = Prob(z_i = m | x_i, theta_t)
        latent_probs = np.array(self._latent_probs(len(data)))
        k = len(data[0])
        for m in xrange(len(self._gmm)):
            latent_sum = np.sum(latent_probs.T[m])
            u_m = 1. / latent_sum * np.dot(data.T, latent_probs.T[m])
            distances = self._gmm.distances(m)
            sigma_m_squared = 1. / latent_sum / k * np.inner(distances, latent_probs.T[m])
            n = len(data)
            c_m = 1. / n * latent_sum
            self._gmm.update_gaussian(m, c_m, Gaussian(u_m, sigma_m_squared ** 0.5))


class GaussianMixtureModel(object):
    """
    Mixture of Gaussians with prior probabilities.

    Pr(x) = SUM(c[m] * Pr(Gaussian[m](x))
    """

    def __init__(self, c_aprior_probs, gaussians):
        """

        :param c_aprior_probs: prior probabilities
        :type c_aprior_probs: list[float]
        :param gaussians:
        :type gaussians: list[Gaussian]
        """
        assert len(gaussians) == len(c_aprior_probs)
        self._c = np.array(c_aprior_probs, dtype=np.float)
        self._gaussians = list(gaussians)
        self._data_length = 0

    def __len__(self):
        return len(self._c)

    def preprocess(self, data):
        """
        Preprocess the given data, allows further calls to be
        meaningfull.

        Must be called before calling any of:
            log_likelihood, weighted_log_likelihoods, distances
        :param data:
        :return:
        """
        for gaussian in self._gaussians:
            gaussian.prepare_distances(data)
        self._data_length = len(data)

    def log_likelihood(self, i):
        """
        CONST + log(Pr(data[i]))
        CONST is constant for all i.
        :param i:
        :return:
        """
        weighted = self.weighted_log_likelihoods(i)
        return scipy.misc.logsumexp(weighted)

    def weighted_log_likelihoods(self, i):
        """
        CONST + [log(Pr(dalta[i], z_i=m)) for m=0...]
        :param i:
        :return:
        """
        return np.log(np.array(self._c)) + np.array([gaussian.log_likelihood(i) for gaussian in self._gaussians])

    def distances(self, m):
        """
        :param m:
        :return: [||x_i - u_m|| ** 2 for i=0...]
        """
        return self._gaussians[m].distances()

    def update_gaussian(self, m, c, gaussian):
        """
        Set the m'th gaussian to

        After calling this function, one needs to preprocess the data again,
            excet for using the function distances(m), using an "m" value that
             its gaussian wasn't changed.
        :param m:
        :param c:
        :param gaussian:
        :return:
        """
        self._c[m] = c
        self._gaussians[m] = gaussian

    def gaussians(self):
        return list(self._gaussians)

    def predictions(self):
        weighted = np.array([self.weighted_log_likelihoods(i) for i in xrange(self._data_length)])
        return np.argmax(weighted, axis=1)


class Gaussian(object):
    """
    Gaussian multivariate with diagonal covvariance matrix
    of the form sigma ** 2 * Id.
    """

    def __init__(self, u, sigma):
        """

        :param u: the expectation vector.
        :type u: np.array
        :param sigma: scalar - the variance of every coordinate.
        """
        self._sigma = sigma
        self._u = u
        self._distances = np.array([], dtype=np.float)

    def prepare_distances(self, data):
        """
        Preprocess the given data.

        Must be called before calling log_likelihood, or distances.
        :param data:
        :return:
        """
        self._distances = np.sum((data - self._u) ** 2, axis=1)
        assert abs(self._distances[1] - distance.norm(data[1] - self._u, ord=2) ** 2) < 0.00001
        assert len(self._distances) == len(data)

    def log_likelihood(self, i):
        assert 0 <= i < len(self._distances)
        k = len(self._u)
        return - k * np.log(self._sigma) - 0.5 / self._sigma ** 2. * self._distances[i]

    def distances(self):
        return self._distances

    def u(self):
        return self._u.copy()


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
    gmm = GaussianMixtureModel([1. / n] * n, [Gaussian(label2x[label], 1.) for label in labels])
    indexed_test_labels = [labels.index(label) for label in mnist_data.test_labels]

    optimizer = GaussianMixtureModelOptimizer(gmm)
    accuracies = []
    log_likelihoods = []
    for j in xrange(20):
        print 'step', j
        optimizer.step(mnist_data.train_data)
        gmm.preprocess(mnist_data.train_data)
        log_likelihood = scipy.misc.logsumexp([gmm.log_likelihood(i) for i in xrange(len(mnist_data.train_data))])
        log_likelihoods.append(log_likelihood)
        gmm.preprocess(mnist_data.test_data)
        accuracies.append(np.sum(gmm.predictions() == np.array(indexed_test_labels, dtype=np.int)) /
                          float(len(mnist_data.test_data)))
        if len(log_likelihoods) > 1:
            improvment = log_likelihoods[-1] - log_likelihoods[-2]
            if improvment < 1e-10:
                break
    return gmm, labels, accuracies, log_likelihoods


def plot_likelihood_scores(output_directory, log_likelihoods):
    plt.cla()
    plt.plot(log_likelihoods, "b*-")
    plt.title("Log likelihood function of iteration")
    plt.xlabel("iteration")
    plt.ylabel("Log-likelihood")
    fname = os.path.join(output_directory, "log_likelihoods.png")
    plt.savefig(fname)
    if SHOW:
        plt.show()


def plot_final_state(output_directory, gmm, labels):
    for label, gaussian in zip(labels, gmm.gaussians()):
        plt.cla()
        plt.imshow(np.reshape(gaussian.u(), (28, 28)))
        plt.title("Digit={}".format(label))
        fname = os.path.join(output_directory, "final_state{:d}.png".format(label))
        plt.savefig(fname)


def plot_accuracy_rates(output_directory, accuracy_rates):
    plt.clf()
    plt.plot(accuracy_rates, "r*-")
    plt.title("Accuracy rate on test set as function of iteration")
    plt.xlabel("iteration")
    plt.ylabel("Accuracy Rate")
    fname = os.path.join(output_directory, "accuracies.png")
    plt.savefig(fname)
    print "Last accuracy:", accuracy_rates[-1]
    if SHOW:
        plt.show()


def main(output_directory):
    final_state, labels, test_accuracy_rates, likelihood_scores = q4()
    plot_likelihood_scores(output_directory, likelihood_scores)
    plot_final_state(output_directory, final_state, labels)
    plot_accuracy_rates(output_directory, test_accuracy_rates)


if __name__ == '__main__':
    main("outputs")
