import unittest
import numpy as np
import q6_linear_multiclass_svm


class TestLinearMulticlassSVM(unittest.TestCase):
    def test_gradient(self):
        C = 1.
        score = q6_linear_multiclass_svm.LinearMulticlassSVM(C)
        d = 4
        k = 2
        m = 5
        w = np.array([[1, 0., 2, 3], [4, 5, 6., 7]])
        assert w.shape == (k, d)
        x = np.array([[1., 2., 3., 4.],
                      [1., 9., 3., 4.],
                      [1., 1., 0., 4.],
                      [1., 7., 3., 12.],
                      [1., 2., 3., 1.]])
        assert x.shape == (m, d)
        y = np.array([0, 0, 0, 1, 1])
        assert y.shape == (m,)
        fx = score.score(w, x, y)
        gradient = np.zeros((k, d))
        for i in xrange(m):
            gradient += score.gradient(w, x[i], y[i])
        gradient /= m
        # Standard gradient test:
        #       (f(x+h) - f(x)) / |h| ~= gradient
        h_size = 2 ** -10.
        for j in xrange(k):
            for t in xrange(len(w[j])):
                h = np.zeros_like(w)
                h[j, t] = h_size
                fx_plus_h = score.score(w + h, x, y)
                gradient_empirical = (fx_plus_h - fx) / h_size
                self.assertAlmostEqual(gradient_empirical, gradient[j, t], places=2)

