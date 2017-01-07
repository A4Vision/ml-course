import itertools
import numpy as np
import collections
import functools
import sys
import os

SHOW = True
import matplotlib
if not SHOW:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import tabulate


def dict2table(d, x_name, y_name, fmt_func=lambda x: x):
    """
    Transform the given dictionary of values, to a table.

    :param d: d[(x, y)] = value.
    :param x_name:
    :param y_name:
    :return:
    >>> d = {('A', 1): "Moshe", ('A', 2.5): "Haim", ('B', 1): "Gabi", ('B', 3): "Juvani"}
    >>> print dict2table(d, "x", "y")
    [[None, 'x=A', 'x=B'], ['y=1', 'Moshe', 'Gabi'], ['y=2.5', 'Haim', None], ['y=3', None, 'Juvani']]
    """
    x_values = sorted(set(key[0] for key in d))
    y_values = sorted(set(key[1] for key in d))
    table = [[None] * len(x_values) for _ in xrange(len(y_values))]
    for (x, y), value in d.iteritems():
        table[y_values.index(y)][x_values.index(x)] = fmt_func(value)
    # Write all the x values as first row.
    first_row = [None] + ['{}={}'.format(x_name, x) for x in x_values]
    table.insert(0, first_row)
    # Write all the y values as first column.
    for i, y in enumerate(y_values):
        table[1 + i].insert(0, '{}={}'.format(y_name, y))
    return table


def validation_dict(measurement_accuracies):
    return {key: measurement.validation for key, measurement in measurement_accuracies.items()}


def train_dict(measurement_accuracies):
    return {key: measurement.train for key, measurement in measurement_accuracies.items()}


AccuracyMeasurement = collections.namedtuple("AccuracyMeasurement", ("validation", "train"))


def best_parameters(measurements_dict):
    best_params, best_measurement = max(measurements_dict.items(), key=lambda (params, measurement): measurement.validation)
    return best_params


def next_parameters_range(arg_parameters, best_parameter, range_size):
    """
    Given a list of parameters, the best choice amongst them,
    returns a list of parameters around the given parametr, of the given size.

    :param arg_parameters: list[float]. The parameters that were already measured.
    :param best_parameter: float. The parameter that gave the best measurement.
    :param range_size: int. size of the output.
    :return: np.array - list of new parameters to measure, around the best parameter.
    """
    assert len(arg_parameters) >= 2
    assert range_size >= 3
    parameters = sorted(arg_parameters)
    middle = best_parameter
    middle_index = np.argmin([abs(x - best_parameter) for x in parameters])
    if middle_index != 0:
        bottom = parameters[middle_index - 1]
    if middle_index != len(parameters) - 1:
        top = parameters[middle_index + 1]
    if middle_index == 0:
        bottom = 2 * middle - top
    if middle_index == len(parameters) - 1:
        top = 2 * middle - bottom
    jump = (top - bottom) / (range_size + 1)
    return np.arange(bottom + jump, top, jump)[:range_size]


def grid_search(function, parameters_dim0, parameters_dim1, search_depth, grid_size):
    """
    :param function:
    :param parameters_dim0:
    :param parameters_dim1:
    :return:
    """
    assert len(parameters_dim0) >= 3
    assert len(parameters_dim1) >= 3
    # Measure over the whole grid.
    accuracy_measurements = {}
    for param0, param1 in itertools.product(parameters_dim0, parameters_dim1):
        accuracy_measurements[(param0, param1)] = function(param0, param1)

    for _ in xrange(search_depth):
        # Find best parameters
        best0, best1 = best_parameters(accuracy_measurements)
        # Build a linear grid around these parameters
        parameters_dim0 = next_parameters_range(parameters_dim0, best0, int(grid_size ** 0.5))
        parameters_dim1 = next_parameters_range(parameters_dim1, best1, int(grid_size ** 0.5))
        for param0, param1 in itertools.product(parameters_dim0, parameters_dim1):
            accuracy_measurements[(param0, param1)] = function(param0, param1)
    return accuracy_measurements


def best_C_eta(measure_wrapper_function, gross_search_iterations, n_gross_search_samples,
               deep_search_iterations, n_deep_search_samples):
    # First, we want a general idea of where to search C and eta.
    etas = [10 ** i for i in xrange(-11, 3)]
    Cs = [10 ** i for i in xrange(-4, 2)]
    measure_accuracies0 = measure_wrapper_function(N=n_gross_search_samples)
    measure0 = functools.partial(measure_accuracies0, iterations=gross_search_iterations)
    # measure0 = lambda x, y: AccuracyMeasurement(validation=-(x - 0.013) ** 2 - (y - 1.2) ** 2, train=0)
    accuracies0 = grid_search(measure0, Cs, etas, 1, 9)

    print tabulate.tabulate(dict2table(validation_dict(accuracies0), "C", "eta", lambda x: "{:.5f}".format(x)))
    # The conclusion:
    C, eta = best_parameters(accuracies0)
    # Second, apply a deep naive grid search.
    Cs = np.arange(C / 3., C * 3, C)
    etas = np.arange(eta / 3., eta * 3, eta)
    measure_accuracies1 = measure_wrapper_function(N=n_deep_search_samples)
    measure1 = functools.partial(measure_accuracies1, iterations=deep_search_iterations)
    # measure1 = measure0
    accuracies1 = grid_search(measure1, Cs, etas, 2, 9)
    print tabulate.tabulate(dict2table(validation_dict(accuracies1), "C", "eta", lambda x: "{:.5f}".format(x)))

    return accuracies0, accuracies1


def generate_error_rate_plots(best_C, best_eta, measure_func, iterations, output_directory):
    graph_Cs = np.arange(best_C / 3., best_C * 3, best_C / 2.)
    validation_error_rates = []
    train_error_rates = []
    for C in graph_Cs:
        measurement = measure_func(eta=best_eta, C=C, iterations=iterations)
        validation_error_rates.append(1 - measurement.validation)
        train_error_rates.append(1 - measurement.train)
    plt.suptitle("Error rates as function of C, using the best eta.")
    plt.xlabel("C")
    plt.ylabel("error rate")
    plt.plot(graph_Cs, validation_error_rates, color='red', label='validation error rate')
    plt.plot(graph_Cs, train_error_rates, color='blue', label='training error rate')

    plt.legend()
    plt.savefig(os.path.join(output_directory, "q6_svm_error_as_function_of_C.png"))
    if SHOW:
        plt.show()

    validation_error_rates = []
    train_error_rates = []
    graph_etas = np.arange(best_eta / 3., best_eta * 3, best_eta / 2.)
    for eta in graph_etas:
        measurement = measure_func(eta=eta, C=best_C, iterations=iterations)
        validation_error_rates.append(1 - measurement.validation)
        train_error_rates.append(1 - measurement.train)


    plt.suptitle("Error rates as function of eta, using the best C.")
    plt.xlabel("eta")
    plt.ylabel("error rate")
    plt.plot(graph_etas, validation_error_rates, color='red', label='validation error rate')
    plt.plot(graph_etas, train_error_rates, color='blue', label='training error rate')
    plt.legend()

    plt.savefig(os.path.join(output_directory, "q6_svm_error_as_function_of_eta.png"))
    if SHOW:
        plt.show()
