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
from posts_reader import postprocess
import classifier as clf
# thierd parties libraries
import pandas as pd


def main():
    """
    """
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--clssifier",
                        dest="classifier",
                        help="Model file",
                        type=str)

    options = parser.parse_args()

    samples = list({
        'I am not fine today',
        'I want to kill you',
        'What a nice day'
    })
    df = pd.Series(samples)

    labels = clf.predict(df, options.classifier)
    print(postprocess(samples, labels))


if __name__ == '__main__':
    main()
