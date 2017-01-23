import os
import time
SHOW = False
import matplotlib
if not SHOW:
    matplotlib.use('Agg')

from matplotlib import pyplot as plt
import math
import data
import numpy as np


def error_rate(prediction, labels):
    assert len(labels) == len(prediction)
    return np.sum(prediction != labels) / float(len(prediction))


class AdaBoostHalfSpaces(object):
    """
    Optimized solution of AdaBoost for images
    with a less than 256 possible values per pixel.
    """
    MAX_PIXEL_OPTIONS = 256

    def __init__(self, train_data, train_labels):
        self._y = train_labels
        self._m = train_data.shape[0]
        self._Dt = np.ones(self._m, dtype=np.float) / self._m
        self._predictor = WeightedHalfSpacesPredictor()
        self.__preprocess(train_data)

    def __preprocess(self, data):
        """
        Convert the data to use indices instead of float values.
        :param data:
        :return:
        """
        print 'Preprocessing...'
        self._x = np.zeros_like(data, dtype=np.uint32)
        self._pixels_options = [None] * self._x.shape[1]
        # for every pixel
        for k in xrange(data.shape[1]):
            pixel_options = sorted(set(data[range(data.shape[0]), k]))
            self._pixels_options[k] = pixel_options
            assert len(pixel_options) < self.MAX_PIXEL_OPTIONS
            for i in xrange(data.shape[0]):
                # We use HalfSpaces, therefore all the information is
                # in the order of the value relative to all posibilities for this pixel.
                self._x[i, k] = pixel_options.index(data[i, k])

    def predictor(self):
        return self._predictor

    def _calculate_g_probs(self):
        """
        Preprocessing for selecting h_t.
        gprobs1[k][i] = SUM_j(Dt[j] ; such that y_j == 1, self._x[j] == i)
        gprobs2[k][i] = SUM_j(Dt[j] ; such that y_j == -1, self._x[j] == i)
        :return: gprobs1, gprobs2
        """
        # 784 X 256 probabilities matrix.
        gprobs1 = np.zeros(shape=(self._x.shape[1], self.MAX_PIXEL_OPTIONS), dtype=np.float)
        gprobs2 = gprobs1.copy()
        for j, (x, y) in enumerate(zip(self._x, self._y)):
            if y > 0:
                arr = gprobs1
            else:
                arr = gprobs2
            arr[range(gprobs1.shape[0]), x] += self._Dt[j]
        return gprobs1, gprobs2

    def _choose_ht(self):
        """
        Argmin(error(h); assuming Prob(x=x_i) = self._Dt[i])
        :return: h_t, epsilon_t
        """
        gprobs1, gprobs2 = self._calculate_g_probs()

        h_t = None
        best_error = 1.
        # For every pixel, find the best halfspace, keep only the global best halfspace.
        for k, (probs1, probs2) in enumerate(zip(gprobs1, gprobs2)):
            # errors[i] = error(h), for h = HalfSpace(threshold=i+1, k_pixel=k, is_positive=True)
            errors = np.cumsum(probs1) + np.sum(probs2) - np.cumsum(probs2)
            # Take the lowest or highest possible error.
            both_sides_errors = 0.5 - np.abs(0.5 - errors)
            i = np.argmin(both_sides_errors)
            error = both_sides_errors[i]

            if error <= best_error:
                positive = bool(errors[i] < 0.5)
                h_t = HalfSpace(i + 1, k, positive)
                best_error = error

        assert h_t is not None
        return h_t, best_error

    def _convert_halfspace(self, h):
        """
        Convert halfspace with theta that referes to an index (in pixel_options[k])
        to an external HalfSpace that one may use with raw data (i.e. not preprocessed).
        :param h:
        :return:
        """
        assert isinstance(h, HalfSpace)
        theta = self._pixels_options[h.k()][h.threshold_theta()]
        return HalfSpace(theta, h.k(), h.positive())

    def step(self):
        """
        AdaBoost standard step.
        :return:
        """
        h_t, epsilon_t = self._choose_ht()
        alpha_t = 0.5 * math.log((1 - epsilon_t) / epsilon_t)
        converted_h_t = self._convert_halfspace(h_t)
        self._predictor.add_weighted_halfspace(alpha_t, converted_h_t)
        ht_applied = h_t.apply(self._x)
        self._Dt *= np.exp(-alpha_t * ht_applied * self._y)
        # Z_t is calculated just for debugging.
        Z_t = 2 * math.sqrt(epsilon_t * (1 - epsilon_t))
        assert abs(np.sum(self._Dt) - Z_t) < 1e-2
        self._Dt /= np.sum(self._Dt)


class WeightedHalfSpacesPredictor(object):
    """
    Predictor composed of T simple HalfSpace predictors.
    predicts: sgn(sum(alpha_j * h_j(x)))
    """

    def __init__(self):
        self._hs = []
        self._alphas = []

    def predict(self, data):
        return np.sign(self._weighted_sum(data))

    def _weighted_sum(self, data):
        weighted_sum = np.zeros(data.shape[0], dtype=np.float)
        for alpha, h in zip(self._alphas, self._hs):
            assert isinstance(h, HalfSpace)
            weighted_sum += alpha * h.apply(data)
        assert not (np.sign(weighted_sum) == 0).any()
        return weighted_sum

    def add_weighted_halfspace(self, alpha, halfspace):
        self._alphas.append(alpha)
        assert isinstance(halfspace, HalfSpace)
        self._hs.append(halfspace)

    def error_rate(self, data, labels):
        return error_rate(self.predict(data), labels)

    def average_exponential_loss(self, data, labels):
        y = labels
        exponential_loss = np.exp(-y * self._weighted_sum(data))
        return np.average(exponential_loss)


class HalfSpace(object):
    """
    Simple HalfSpace predictor:
        if positive:
            predicts x[k] >= threshold_theta
        otherwise:
            predicts x[k] < threshold_theta
    """

    def __init__(self, threshold_theta, k_pixel, is_positive):
        assert isinstance(is_positive, bool)
        assert isinstance(k_pixel, int)
        self._k = k_pixel
        self._threshold_theta = threshold_theta
        self._is_positive = is_positive

    def apply(self, x_data):
        assert x_data.ndim == 2
        is_larger_than_i = (x_data[range(x_data.shape[0]), self._k] >= self._threshold_theta)
        positive_result = 2 * np.int32(is_larger_than_i) - 1
        if self._is_positive:
            return positive_result
        else:
            return -1 * positive_result

    def apply_single(self, x):
        assert x.ndim == 1
        if x[self._k] >= self._threshold_theta:
            if self._is_positive:
                return 1
            else:
                return -1
        else:
            if self._is_positive:
                return -1
            else:
                return 1

    def threshold_theta(self):
        return self._threshold_theta

    def positive(self):
        return self._is_positive

    def k(self):
        return self._k


def main(output_directory):
    start = time.time()
    boost = AdaBoostHalfSpaces(data.train_data, data.train_labels)
    T = 80
    train_error_rates = []
    test_error_rates = []
    train_exp_losses = []
    test_exp_losses = []
    for i in xrange(T):
        print 'i=', i
        boost.step()
        predictor = boost.predictor()
        train_error = predictor.error_rate(data.train_data, data.train_labels)
        test_error = predictor.error_rate(data.test_data, data.test_labels)
        train_avg_exp_loss = predictor.average_exponential_loss(data.train_data, data.train_labels)
        test_avg_exp_loss = predictor.average_exponential_loss(data.test_data, data.test_labels)
        train_error_rates.append(train_error)
        test_error_rates.append(test_error)
        train_exp_losses.append(train_avg_exp_loss)
        test_exp_losses.append(test_avg_exp_loss)
    plt.figure(figsize=(10, 8))
    plt.title("Error rate as function of iteration")
    plt.xlabel("iteration")
    plt.xlabel("ErroRate")
    plt.plot(train_error_rates, 'g-*', label="Training Error Rate")
    plt.plot(test_error_rates, 'r--', label="Test Error Rate")
    plt.savefig(os.path.join(output_directory, "error_rate.png"))
    if SHOW:
        plt.show()
    plt.cla()
    plt.title("Average Exponential Loss: AVERAGE_i(exp(- y_i * SUM_j{alpha_j * h_j(x_i)}))")
    plt.xlabel("iteration (T)")
    plt.xlabel("Average Exponential Loss")
    plt.plot(train_exp_losses, 'g-*', label="Training Average Exponential Loss")
    plt.plot(test_exp_losses, 'r--', label="Test Average Exponential Loss")
    plt.savefig(os.path.join(output_directory, "average_exponential_loss.png"))
    if SHOW:
        plt.show()
    print 'Total RunTime', time.time() - start


if __name__ == '__main__':
    main("outputs")
