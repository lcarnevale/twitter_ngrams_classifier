#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

.. _Google Python Style Guide
    https://github.com/google/styleguide/blob/gh-pages/pyguide.md
"""

__copyright__ = 'Copyright 2019, University of Messina'
__author__ = 'Lorenzo Carnevale <lorenzocarnevale@gmail.com>
__credits__ = ''
__description__ = ''

# standard libraries
import json
import warnings
import argparse
# local libraries
from posts_reader import PostsReader
from models.incr import Classifier as IncrClassifier
from models.incr2step import Classifier as Incr2StepClassifier
from models.notincr import Classifier as NotIncrClassifier



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

    tests = [1,2,3,4,5,6,7,8]
    posts = PostsReader(postset).process()

    print('Incremental Dataset')
    incr_clf = IncrClassifier(posts, tests)
    incr_clf.run()
    incr_clf.export()
    print('Incremental 2 Steps Dataset')
    incr2step_clf = Incr2StepClassifier(posts, tests)
    incr2step_clf.run()
    incr2step_clf.export()
    print('Not Incremental Dataset')
    notincr_clf = NotIncrClassifier(posts, tests)
    notincr_clf.run()
    notincr_clf.export()


if __name__ == '__main__':
    main()
