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
import classifier as cls
# from incr import Classifier as IncrClassifier
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

    parser.add_argument("-max", "--ngrammax",
                        dest="ngram_max",
                        help="Maximum number of ngrams",
                        type=int)

    parser.add_argument("-min", "--ngrammin",
                        dest="ngram_min",
                        help="Minimum number of ngrams",
                        type=int)

    options = parser.parse_args()


    with open(options.postset) as f:
        postset = json.load(f)
    df = pd.read_json(options.postset, orient='columns')
    df.columns = ['samples']

    posts = preprocess(df['samples'])
    cls.train(posts, options.ngram_min, options.ngram_max)

if __name__ == '__main__':
    main()
