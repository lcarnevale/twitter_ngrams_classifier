#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Application.

.. _Google Python Style Guide
    https://github.com/google/styleguide/blob/gh-pages/pyguide.md
"""

__copyright__ = 'Copyright 2019, University of Messina'
__author__ = 'Lorenzo Carnevale <lorenzocarnevale@gmail.com>'
__credits__ = ''
__description__ = 'Incremental Analysis'

# standard libraries
import os
import json
import warnings
import argparse
import collections
# local libraries
import classifier as cls
from posts_reader import preprocess
# thierd parties libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

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

    options = parser.parse_args()

    if not options.verbose:
        warnings.filterwarnings("ignore")

    # todo: replace with read_csv
    with open(options.postset) as f:
        postset = json.load(f)
    df = pd.read_json(options.postset, orient='columns')
    df.columns = ['samples']

    posts = preprocess(df['samples'])

    scores = list()
    max_ = options.ngram_min
    while max_ <= options.ngram_max:
        ngrams = [ i for i in range(options.ngram_min, max_) ]
        print('Analysis with %s ngrams' % (ngrams))
        score = cls.train(posts, options.ngram_min, max_)
        scores.append(flatten(score, sep='.'))
        max_ +=  1
    df = pd.DataFrame(scores)
    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    with open("scores-incr.json", 'w') as f:
        json.dump(scores, f)

    # Naive Bayes accuracy comparison
    plt.figure()
    df.plot(x='ngrams', y=['scores.BernoulliNB.accuracy', 'scores.MultinomialNB.accuracy'], title='SVC Accuracy - Incremental', legend=True)
    plt.savefig('../results/naive_accuracy_incr.png')
    # Naive Bayes precision comparison
    plt.figure()
    df.plot(x='ngrams', y=['scores.BernoulliNB.precision.notalarm', 'scores.BernoulliNB.precision.alarm', 'scores.BernoulliNB.precision.suspect', 'scores.MultinomialNB.precision.notalarm', 'scores.MultinomialNB.precision.alarm', 'scores.MultinomialNB.precision.suspect'], title='Naive Bayes Precision - Incremental', legend=True)
    plt.savefig('../results/naive_precision_incr.png')
    # Naive Bayes recall comparison
    plt.figure()
    df.plot(x='ngrams', y=['scores.BernoulliNB.recall.notalarm', 'scores.BernoulliNB.recall.alarm', 'scores.BernoulliNB.recall.suspect', 'scores.MultinomialNB.recall.notalarm', 'scores.MultinomialNB.recall.alarm', 'scores.MultinomialNB.recall.suspect'], title='Naive Bayes Recall - Incremental', legend=True)
    plt.savefig('../results/naive_recall_incr.png')
    # Naive Bayes f1score comparison
    plt.figure()
    df.plot(x='ngrams', y=['scores.BernoulliNB.f1score.notalarm', 'scores.BernoulliNB.f1score.alarm', 'scores.BernoulliNB.f1score.suspect', 'scores.MultinomialNB.f1score.notalarm', 'scores.MultinomialNB.f1score.alarm', 'scores.MultinomialNB.f1score.suspect'], title='Naive Bayes F1Score - Incremental', legend=True)
    plt.savefig('../results/naive_f1score_incr.png')
    # Linear accuracy comparison
    plt.figure()
    df.plot(x='ngrams', y=['scores.LogisticRegression.accuracy', 'scores.SGDClassifier.accuracy'], title='SVC Accuracy - Incremental', legend=True)
    plt.savefig('../results/linear_accuracy_incr.png')
    # Linear precision comparison
    plt.figure()
    df.plot(x='ngrams', y=['scores.LogisticRegression.precision.notalarm', 'scores.LogisticRegression.precision.alarm', 'scores.LogisticRegression.precision.suspect', 'scores.SGDClassifier.precision.notalarm', 'scores.SGDClassifier.precision.alarm', 'scores.SGDClassifier.precision.suspect'], title='Linear Precision - Incremental', legend=True)
    plt.savefig('../results/linear_precision_incr.png')
    # Linear recall comparison
    plt.figure()
    df.plot(x='ngrams', y=['scores.LogisticRegression.recall.notalarm', 'scores.LogisticRegression.recall.alarm', 'scores.LogisticRegression.recall.suspect', 'scores.SGDClassifier.recall.notalarm', 'scores.SGDClassifier.recall.alarm', 'scores.SGDClassifier.recall.suspect'], title='Linear Recall - Incremental', legend=True)
    plt.savefig('../results/linear_recall_incr.png')
    # Linear f1score comparison
    plt.figure()
    df.plot(x='ngrams', y=['scores.LogisticRegression.f1score.notalarm', 'scores.LogisticRegression.f1score.alarm', 'scores.LogisticRegression.f1score.suspect', 'scores.SGDClassifier.f1score.notalarm', 'scores.SGDClassifier.f1score.alarm', 'scores.SGDClassifier.f1score.suspect'], title='Linear F1Score - Incremental', legend=True)
    plt.savefig('../results/linear_f1score_incr.png')
    # SVC accuracy comparison
    plt.figure()
    df.plot(x='ngrams', y=['scores.SVC.accuracy', 'scores.LinearSVC.accuracy'], title='SVC Accuracy - Incremental', legend=True)
    plt.savefig('../results/svc_accuracy_incr.png')
    # SVC precision comparison
    plt.figure()
    df.plot(x='ngrams', y=['scores.SVC.precision.notalarm', 'scores.SVC.precision.alarm', 'scores.SVC.precision.suspect', 'scores.LinearSVC.precision.notalarm', 'scores.LinearSVC.precision.alarm', 'scores.LinearSVC.precision.suspect'], title='SVC Precision - Incremental', legend=True)
    plt.savefig('../results/svc_precision_incr.png')
    # SVC recall comparison
    plt.figure()
    df.plot(x='ngrams', y=['scores.SVC.recall.notalarm', 'scores.SVC.precision.alarm', 'scores.SVC.recall.suspect', 'scores.LinearSVC.recall.notalarm', 'scores.LinearSVC.recall.alarm', 'scores.LinearSVC.recall.suspect'], title='SVC Recall - Incremental', legend=True)
    plt.savefig('../results/svc_recall_incr.png')
    # SVC f1score comparison
    plt.figure()
    df.plot(x='ngrams', y=['scores.SVC.f1score.notalarm', 'scores.SVC.f1score.alarm', 'scores.SVC.f1score.suspect', 'scores.LinearSVC.f1score.notalarm', 'scores.LinearSVC.f1score.alarm', 'scores.LinearSVC.f1score.suspect'], title='SVC F1Score - Incremental', legend=True)
    plt.savefig('../results/svc_f1score_incr.png')


if __name__ == '__main__':
    main()
