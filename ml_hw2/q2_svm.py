import math
import random
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
SHOW = True
import tabulate
import os
import matplotlib
if not SHOW:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
import mnist_data
from sklearn import svm


class OneOffTrainedSVM(object):
    def __init__(self, C, training_data, training_labels):
        self._classifier = svm.LinearSVC(loss="hinge", fit_intercept=False, C=C)
        self._classifier.fit(training_data, training_labels)

    def accuracy(self, data, labels):
        return np.sum(self._classifier.predict(data) == labels) / float(len(data))

    def get_w(self):
        return self._classifier.coef_


def generate_C_values(middle, bottom, top, amount):
    return np.arange(bottom, top, (top - bottom) / amount)


def accuracies_for_C_values(c_values, training_data, training_labels, validation_data, validation_labels):
    classifiers = [OneOffTrainedSVM(c, training_data, training_labels) for c in c_values]
    return [(classifier.accuracy(validation_data, validation_labels),
             classifier.accuracy(training_data, training_labels)) for classifier in classifiers]


def optimize_C(training_data, training_labels, validation_data, validation_labels):
    c_values = [3 ** i for i in xrange(-5, 5)]
    all_validation_accuracies = {}
    all_training_accuracies = {}
    for i in xrange(4):
        accuracies = accuracies_for_C_values(c_values, training_data, training_labels, validation_data, validation_labels)
        best_c_index = np.argmax([validation_accuracy for validation_accuracy, _ in accuracies])
        print 'accuracies', accuracies
        best_c = c_values[best_c_index]
        bottom = c_values[max(0, best_c_index - 2)]
        top = c_values[min(len(c_values) - 1, best_c_index + 2)]
        for c, (validation_accuracy, training_accuracy) in zip(c_values, accuracies):
            all_validation_accuracies[c] = validation_accuracy
            all_training_accuracies[c] = training_accuracy
        c_values = generate_C_values(best_c, bottom, top, 5)
    return all_validation_accuracies, all_training_accuracies


def q2(output_directory, normalized_train_data, normalized_validation_data, normalized_test_data):
    print "Q2a"
    validation_accuracies, training_accuracies = optimize_C(normalized_train_data, mnist_data.train_labels,
                                                            normalized_validation_data, mnist_data.validation_labels)
    best_c = max(validation_accuracies.items(), key=lambda (c, accuracy): accuracy)[0]
    c_values = sorted(validation_accuracies.keys())
    validation_accuracies_arr = [validation_accuracies[c] for c in c_values]
    training_accuracies_arr = [training_accuracies[c] for c in c_values]

    plt.plot(c_values, validation_accuracies_arr, color="blue", marker="*", label="Validation Accuracy")
    plt.plot(c_values, training_accuracies_arr, color="red", marker="+", label="Training Accuracy")
    plt.xlabel("C = margin constant for SVM")
    plt.ylabel("Accuracy")
    plt.legend(loc=5)
    plt.suptitle("SKLearn linear SVM Accuracy as function of C")
    plt.savefig(os.path.join(output_directory, "Q2a_SVMAccuracy.png"))
    if SHOW:
        plt.show()
    plt.cla()
    print "best C:", best_c
    print "Q2b"
    best_classifier = OneOffTrainedSVM(best_c, normalized_train_data, mnist_data.train_labels)
    print "Q2c"
    w = best_classifier.get_w()
    outfile = os.path.join(output_directory, "Q2c_SVMWeights.png")
    plt.imsave(outfile, np.reshape(w, (28, 28)))
    if SHOW:
        plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
        plt.show()
    plt.cla()

    print "Q2d"
    print "Best calssifier accuracy:", best_classifier.accuracy(normalized_test_data, mnist_data.test_labels)


def main(dir_name):
    # # Normalize the data - ||x|| == 1
    normalized_train_data = preprocessing.normalize(mnist_data.train_data, axis=1)
    normalized_test_data = preprocessing.normalize(mnist_data.test_data, axis=1)
    normalized_validation_data = preprocessing.normalize(mnist_data.validation_data, axis=1)
    q2(dir_name, normalized_train_data, normalized_validation_data, normalized_test_data)


if __name__ == '__main__':
    dir_name = os.path.join(os.path.dirname(__file__), "outputs")
    main(dir_name)
