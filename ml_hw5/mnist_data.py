from numpy import *
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

train_idx = numpy.random.RandomState(0).permutation(range(60000))

train_data_size = 1000

train_data = data[train_idx[:train_data_size], :].astype(float)
train_labels = labels[train_idx[:train_data_size]]

train_subset = where((train_labels == 0) | (train_labels == 1) | (train_labels == 3) | (train_labels == 4) | (train_labels == 8))[0]
train_data = train_data[train_subset, :]
train_labels = train_labels[train_subset]

test_data = data[60000:, :].astype(float)
test_labels = labels[60000:]

test_subset = where((test_labels == 0) | (test_labels == 1) | (test_labels == 3) | (test_labels == 4) | (test_labels == 8))[0]
test_data = test_data[test_subset, :]
test_labels = test_labels[test_subset]