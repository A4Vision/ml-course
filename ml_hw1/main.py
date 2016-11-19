#!/usr/bin/python2.7
import sys
import os
# Allow any further imports to happen smoothly.
sys.path.insert(0, os.path.dirname(__file__))
import q1_knn
import q2_union_of_intervals


def main(argv):
    if len(argv) != 2:
        print "Usage: {} <output-directory>".format(__file__)
        sys.exit(2)
    output_directory = argv[1]
    os.mkdir(output_directory)
    q1_knn.main(output_directory)
    q2_union_of_intervals.main(output_directory)


if __name__ == '__main__':
    main(sys.argv)