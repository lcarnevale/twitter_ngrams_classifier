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
import os
import json
import warnings
import argparse
# local libraries
from posts_reader import preprocess
import classifier as cls
# thierd parties libraries
import pandas as pd
import matplotlib.pyplot as plt


# def evalutation():
#     """
#     """
#     if not os.path.exists('../results'):
#         os.makedirs('../results')
#     # save on csv
#     with open('../results/accuracy_incr.csv', 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(self.tests)
#         # writer.writerow(self.vocabulary_sizes)
#         for setting in self.__settings:
#             writer.writerow(setting['scores']['accuracy'])
#
#     with open('../results/precision_incr.csv', 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(self.tests)
#         for setting in self.__settings:
#             row = [ score for elem in setting['scores']['precision'] for score in elem.values() ]
#             writer.writerow(row)
#
#     with open('../results/recall_incr.csv', 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(self.tests)
#         for setting in self.__settings:
#             row = [ score for elem in setting['scores']['recall'] for score in elem.values() ]
#             writer.writerow(row)
#
#     with open('../results/f1score_incr.csv', 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(self.tests)
#         for setting in self.__settings:
#             row = [ score for elem in setting['scores']['f1score'] for score in elem.values() ]
#             writer.writerow(row)

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

    # parser.add_argument("-a", "--approach",
    #                     dest="type",
    #                     help="Select one of the available approaches: incremental (incr); incremental 2 steps (incr2step); not incremental (notincr)",
    #                     type=str, default='incr')

    options = parser.parse_args()

    if not options.verbose:
        warnings.filterwarnings("ignore")

    # todo: replace with read_csv
    with open(options.postset) as f:
        postset = json.load(f)
    df = pd.read_json(options.postset, orient='columns')
    df.columns = ['samples']

    posts = preprocess(df['samples'])

    import collections

    def flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


    scores = list()
    max_ = options.ngram_min + 1
    while max_ <= options.ngram_max:
        ngrams = [ i for i in range(options.ngram_min, max_) ]
        print('Analysis with %s ngrams' % (ngrams))
        score = cls.train(posts, options.ngram_min, max_)
        scores.append(flatten(score, sep='.'))
        max_ +=  1
    df = pd.DataFrame(scores)
    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    # Naive Bayes accuracy comparison
    df.plot(x='ngrams', y=['scores.BernoulliNB.accuracy', 'scores.MultinomialNB.accuracy'], legend=True)
    plt.savefig('../results/naive_accuracy.png')
    # Linear accuracy comparison
    df.plot(x='ngrams', y=['scores.LogisticRegression.accuracy', 'scores.SGDClassifier.accuracy'], legend=True)
    plt.savefig('../results/linear_accuracy.png')
    # SVC accuracy comparison
    df.plot(x='ngrams', y=['scores.SVC.accuracy', 'scores.LinearSVC.accuracy'], legend=True)
    plt.savefig('../results/svc_accuracy.png')
    # Naive Bayes accuracy comparison
    df.plot(x='ngrams', y=['scores.BernoulliNB.precision.notalarm', 'scores.BernoulliNB.precision.alarm', 'scores.BernoulliNB.precision.suspect', 'scores.MultinomialNB.precision.notalarm', 'scores.MultinomialNB.precision.alarm', 'scores.MultinomialNB.precision.suspect'], legend=True)
    plt.savefig('../results/naive_precision.png')



if __name__ == '__main__':
    main()
