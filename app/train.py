#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Application.

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
from posts_reader import preprocess
from incr import Classifier as IncrClassifier
# from models.incr2step import Classifier as Incr2StepClassifier
# from models.notincr import Classifier as NotIncrClassifier
# thierd parties libraries
import pandas as pd

def main():
    """
    """
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--postset",
                        dest="postset",
                        help="Dataset",
                        type=str)

    options = parser.parse_args()


    with open(options.postset) as f:
        postset = json.load(f)
    df = pd.read_json(options.postset, orient='columns')
    # View the first ten rows
    df.head()

    # posts = preprocess(postset)

    # tests = [2,3,4,5]
    # print('Incremental Dataset')
    # incr_clf = IncrClassifier(posts, tests)
    # incr_clf.train()
    # incr_clf.export()
    # print('Incremental 2 Steps Dataset')
    # incr2step_clf = Incr2StepClassifier(posts, tests)
    # incr2step_clf.run()
    # incr2step_clf.export()
    # print('Not Incremental Dataset')
    # notincr_clf = NotIncrClassifier(posts, tests)
    # notincr_clf.run()
    # notincr_clf.export()


if __name__ == '__main__':
    main()
