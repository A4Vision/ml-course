import os

import numpy
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

import intervals
import interval_set


def Bounds2IntervalSet(intervals_bounds):
    intervals_list = [interval_set.Interval(float(lower), float(upper)) for lower, upper in intervals_bounds]
    return interval_set.IntervalSet(intervals_list)


def IntervalsSetTotalLength(interval_set):
    return sum([i.upper_bound - i.lower_bound for i in interval_set.intervals], 0)


ONES_INTERVAL_SET = Bounds2IntervalSet([(0, 0.25), (0.5, 0.75)])
ZERO_INTERVAL_SET = Bounds2IntervalSet([(0, 1)]).difference(ONES_INTERVAL_SET)
PROB_ONE = 0.8
PROB_ZERO = 0.1

def RandomDSorted(m):
    x_values = numpy.random.uniform(0, 1, size=m)
    y_values = []
    for x in x_values:
        if x in ONES_INTERVAL_SET:
            p = PROB_ONE
        else:
            p = PROB_ZERO
        y_values.append(int(numpy.random.random() < p))

    y = numpy.array(y_values)[numpy.argsort(x_values)]
    x = numpy.sort(x_values)
    return x, y


def PredictY(x_values, intervals):
    result = numpy.zeros(x_values.shape, dtype=numpy.int32)
    for interval in intervals:
        flags_in_interval = (interval[0] < x_values) & (x_values < interval[1])
        result |= flags_in_interval
    return result


def EmpiricalError(x_values, y_values, intervals):
    calculated_ys = PredictY(x_values, intervals)
    return (calculated_ys != y_values).sum() / float(x_values.size)


def TrueError(intervals_bounds):
    ones = Bounds2IntervalSet(intervals_bounds)
    zeros = Bounds2IntervalSet([(0, 1)]).difference(ones)
    # Prob 0.8 for 1, and the predictor says it should be 1 always.
    length_11 = IntervalsSetTotalLength(ONES_INTERVAL_SET.intersection(ones))
    # Prob 0.8 for 1, and the predictor says it should be 0 always.
    length_10 = IntervalsSetTotalLength(ONES_INTERVAL_SET.intersection(zeros))
    # Prob 0.1 for 1, and the predictor says it should be 1 always.
    length_01 = IntervalsSetTotalLength(ZERO_INTERVAL_SET.intersection(ones))
    # Prob 0.1 for 1, and the predictor says it should be 0 always.
    length_00 = IntervalsSetTotalLength(ZERO_INTERVAL_SET.intersection(zeros))
    assert abs(-1 + length_00 + length_01 + length_10 + length_11) < 0.001
    error = length_11 * (1 - PROB_ONE) + length_10 * PROB_ONE + length_01 * (1 - PROB_ZERO) + length_00 * PROB_ZERO
    return error


def Q2_a(output_directory):
    print "::Question 2a::"
    m = 100
    x, y = RandomDSorted(m)

    def part_i():
        pyplot.gca().set_ylim([-0.1, 1.1])
        pyplot.plot(x, y, 'b^', label="samples")
    def part_ii():
        for x_boundary in [0.25, 0.5, 0.75]:
            pyplot.plot([x_boundary, x_boundary], [-0.1, 1.1], 'r-')
    def part_iii():
        k_intervals, error = intervals.find_best_interval(x, y, 2)
        for x0, x1 in k_intervals:
            pyplot.plot([x0, x1], [0.5, 0.5], 'g-', linewidth=4, label="ERM interval")
    part_i()
    part_ii()
    part_iii()
    pyplot.legend()
    pyplot.xlabel("x ({} uniform samples)".format(m))
    pyplot.ylabel("y (label - 0/1)")
    pyplot.title("Question 2a:: Random samples from distribution,\nand predicted intervals by ERM(k=2)")
    pyplot.savefig(os.path.join(output_directory, "Q2a.png"))
    pyplot.clf()


def Q2_c(output_directory):
    print "::Question 2c::"
    T = 100
    k = 2
    m_values = range(10, 101)
    average_true_errors = []
    average_empricial_errors = []
    for m in m_values:
        total_true_error = 0
        total_empirical_error = 0
        for i in xrange(T):
            # part i
            x, y = RandomDSorted(m)
            k_intervals, best_error = intervals.find_best_interval(x, y, k)
            # part ii
            empirical_error = best_error / float(m) #EmpiricalError(x, y, k_intervals)
            total_empirical_error += empirical_error
            assert round(empirical_error * m) == best_error
            # part iii
            true_error = TrueError(k_intervals)
            total_true_error += true_error
        average_empricial_errors.append(total_empirical_error / T)
        average_true_errors.append(total_true_error / T)
    pyplot.plot(m_values, average_empricial_errors, 'ro--', label="empirical error")
    pyplot.plot(m_values, average_true_errors, 'b^--', label="true error")
    pyplot.legend()
    pyplot.xlabel("m (samples amount)")
    pyplot.ylabel("Average error")
    pyplot.title("Question 2c:: Average error of ERM using k=2 intervals,\nas a function of samples amount")
    pyplot.savefig(os.path.join(output_directory, "Q2c.png"))
    pyplot.clf()


def Q2_d(output_directory):
    print "::Question 2d::"
    m = 50
    k_values = range(1, 21)
    errors = []
    x, y = RandomDSorted(m)
    for k in k_values:
        k_intervals, error = intervals.find_best_interval(x, y, k)
        errors.append(error / float(m))
    best_k = k_values[numpy.argmin(errors)]
    print "\tBest k:", best_k
    pyplot.plot(k_values, errors, 'b^--')
    pyplot.title("Quetion 2d:: Empirical error as a function of k (using ERM)")
    pyplot.xlabel("k")
    pyplot.ylabel("Empirical error")
    pyplot.ylim([min(errors) - 0.01, max(errors) + 0.01])
    pyplot.savefig(os.path.join(output_directory, "Q2d.png"))
    pyplot.clf()


def Q2_e(output_directory):
    print "::Question 2e::"
    T = 4
    m = 50
    k_values = range(1, 21)
    errors_total = numpy.zeros((len(k_values),), dtype=numpy.float32)
    for i in xrange(T):
        errors = []
        x, y = RandomDSorted(m)
        for k in k_values:
            k_intervals, error = intervals.find_best_interval(x, y, k)
            errors.append(error / float(m))
        errors_total += numpy.array(errors)
    errors_average = errors_total / T
    pyplot.plot(k_values, errors_average, 'b^--')
    pyplot.title("Question 2e:: Average empirical error as a function of k (using ERM)")
    pyplot.xlabel("k")
    pyplot.ylabel("Average empirical error")
    pyplot.ylim([min(errors_average) - 0.01, max(errors_average) + 0.01])
    pyplot.savefig(os.path.join(output_directory, "Q2e.png"))
    pyplot.clf()


def Q2_f(output_directory):
    print "::Question 2f::"
    m = 50
    k_values = range(1, 21)
    errors = []
    x, y = RandomDSorted(m)
    validation_x, validation_y = RandomDSorted(m)
    for k in k_values:
        k_intervals, error = intervals.find_best_interval(x, y, k)
        validation_error = EmpiricalError(validation_x, validation_y, k_intervals)
        errors.append(validation_error)
        if k == 6:
            pyplot.plot(x, y, 'ro')
            pyplot.plot(validation_x, validation_y, 'b*')
    best_k = k_values[numpy.argmin(errors)]
    print "\tBest k using validation:", best_k
    pyplot.plot([best_k, best_k], [0, 1], "r--", label="best k")
    pyplot.plot(k_values, errors, 'b^--', label="validation empirical error")
    pyplot.legend()
    pyplot.title("Quetion 2f:: Empirical error of validation\nas a function of k (using ERM)")
    pyplot.xlabel("k")
    pyplot.ylabel("Empirical error")
    pyplot.ylim([min(errors) - 0.01, max(errors) + 0.01])
    pyplot.savefig(os.path.join(output_directory, "Q2f.png"))
    pyplot.clf()


def main(output_directory):
    numpy.random.seed(1234)
    start_a = time.time()
    Q2_a(output_directory)
    print "2a runtime:", time.time() - start_a
    start_c = time.time()
    Q2_c(output_directory)
    print "2c runtime:", time.time() - start_c
    start_d = time.time()
    Q2_d(output_directory)
    print "2d runtime:", time.time() - start_d
    start_e = time.time()
    Q2_e(output_directory)
    print "2e runtime:", time.time() - start_e
    start_f = time.time()
    Q2_f(output_directory)
    print "2f runtime:", time.time() - start_f


