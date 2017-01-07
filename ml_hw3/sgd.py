import numpy as np


class ScoreCalculator(object):
    """
    The score function -
        Calculates the score,
        Calculates the gradient/stochastic gradient.
    """

    def gradient(self, w, x_i, y_i):
        raise NotImplementedError

    def score(self, w, x, y):
        raise NotImplementedError


class StochasticGradientDescent(object):
    """
    Implements (stochastic) gradient descent:
        x -= eta * gradient(x)
    """
    def __init__(self, w0, eta, gradient_calculator):
        assert isinstance(gradient_calculator, ScoreCalculator)
        self._w = w0
        self._gradient_calculator = gradient_calculator
        self._eta = eta

    def step(self, x_i, y_i):
        gradient = self._gradient_calculator.gradient(self._w, x_i, y_i)
        self._w -= self._eta * gradient

    def w(self):
        return self._w.copy()

    def run(self, n_steps, x, y):
        permutation = np.random.permutation(len(x))
        x = x[permutation]
        y = y[permutation]
        for i in xrange(n_steps):
            i_actual = i % len(x)
            self.step(x[i_actual], y[i_actual])

    def score(self, x, y):
        return self._gradient_calculator.score(self._w, x, y)

