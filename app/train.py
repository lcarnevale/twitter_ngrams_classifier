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
# thierd parties libraries
import pandas as pd


def main():
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        help="Increase verbosity",
                        action='store_true')

    parser.add_argument("-d", "--postset",
                        dest="postset",
                        help="Dataset file path. A CSV file is required.",
                        type=str)

    parser.add_argument("-min", "--ngrammin",
                        dest="ngram_min",
                        help="Minimum number of ngrams",
                        type=int)

    parser.add_argument("-max", "--ngrammax",
                        dest="ngram_max",
                        help="Maximum number of ngrams",
                        type=int)

    parser.add_argument("-s", "--save",
                        dest="save",
                        help="Save",
                        action='store_true')

    options = parser.parse_args()

    if not options.verbose:
        warnings.filterwarnings("ignore")

    with open(options.postset) as f:
        postset = json.load(f)
    df = pd.read_json(options.postset, orient='columns')
    df.columns = ['samples']

    posts = preprocess(df['samples'])
    scores = cls.train(posts, options.ngram_min, options.ngram_max, save=options.save)
    print(scores)

if __name__ == '__main__':
    main()
