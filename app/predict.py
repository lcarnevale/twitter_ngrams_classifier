#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

.. _Google Python Style Guide
    https://github.com/google/styleguide/blob/gh-pages/pyguide.md
"""

__copyright__ = 'Copyright 2019, University of Messina'
__author__ = 'Lorenzo Carnevale <lorenzocarnevale@gmail.com>'
__credits__ = ''
__description__ = ''

# standard libraries
import json
import warnings
import argparse
# local libraries
from posts_reader import PostsReader
from incr import Classifier as IncrClassifier
# from incr2step import Classifier as Incr2StepClassifier
# from notincr import Classifier as NotIncrClassifier


def main():
    """
    """
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    options = parser.parse_args()

    reader = PostsReader([
        'I am not fine today',
        'I want to kill you',
        'What a nice day'
    ])
    samples = reader.preprocess()

    clf = IncrClassifier()
    labels = clf.predict(samples)

    result = reader.postprocess(samples, labels)
    print(result)


if __name__ == '__main__':
    main()
