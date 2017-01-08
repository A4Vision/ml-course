#!/usr/bin/python2.7
import sys
import os
# Allow any further imports to happen smoothly.
sys.path.insert(0, os.path.dirname(__file__))
import q6_linear_multiclass_svm
import q7_kernel_multiclass_svm

def main(argv):
    if len(argv) != 2:
        print "Usage: {} <output-directory>".format(__file__)
        sys.exit(2)
    output_directory = argv[1]
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    assert os.path.isdir(output_directory)
    q6_linear_multiclass_svm.main(output_directory)
    q7_kernel_multiclass_svm.main(output_directory)


if __name__ == '__main__':
    main(sys.argv)
