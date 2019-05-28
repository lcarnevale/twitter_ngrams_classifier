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
import os
import csv
import pickle
# local libraries
from metrics import precision_recall
from metrics import f1score
# third parties libraries
import pandas as pd
from nltk import ngrams
from nltk import FreqDist
from nltk.classify import accuracy
from nltk.classify import SklearnClassifier
from textblob import TextBlob
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


classifiers = {
        "BernoulliNB": BernoulliNB(),
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(),
        "SGDClassifier": SGDClassifier(),
        "SVC": SVC(),
        "LinearSVC": LinearSVC()
}


def __generate_features(samples_tokenized, ngram_min, ngram_max):
    """
    """
    features = list()
    for n in range(ngram_min, ngram_max):
        for sample in samples_tokenized:
            for sample_ngrammed in ngrams(sample, n):
                features.append(sample_ngrammed)

    features = pd.Series(features)
    dataset = pd.DataFrame(features, columns=['features'])
    def process(feature):
        blob = TextBlob(" ".join(feature))
        polarity = blob.sentences[0].sentiment.polarity
        if polarity <= -0.3:
            return "alarm"
        elif polarity >= 0.3:
            return "notalarm"
        else:
            return "suspect"
    dataset['classes'] = dataset['features'].apply(process)
    dataset = dataset.drop_duplicates('features').reset_index(drop=True)
    dataset['features'] = dataset['features'].apply(FreqDist)
    dataset = dataset.set_index('features')
    return dataset.to_records().tolist()

def __scores(clf, testset):
    """
    """
    accuracy_ = accuracy(clf, testset)
    precision_, recall_ = precision_recall(clf, testset)
    f1score_ = f1score(precision_, recall_)
    return accuracy_, precision_, recall_, f1score_

def train(samples_tokenized, ngram_min, ngram_max, save=None):
    """
    """
    info = dict()
    # info['ngrams'] = [ i for i in range(ngram_min, ngram_max) ]
    info['ngrams'] = ngram_max -1


    # print('Samples size %s' % (len(samples_tokenized)))
    # print('Generate features...')
    dataset = __generate_features(samples_tokenized, ngram_min, ngram_max)
    if save:
        # todo: implement it
        pass
    # print('Generate features...completed')

    trainset, testset = train_test_split(dataset, test_size=0.33, random_state=42)
    # print('Vocabulary size %s, trainset %s, testset %s' % (len(dataset),len(trainset),len(testset)))
    info['vocabulary_sizes'] = len(dataset)
    info['trainset_sizes'] = len(trainset)
    info['testset_sizes'] = len(testset)
    info['scores'] = dict()

    for classifier in classifiers.keys():
        # print('%s classifier...' % (classifier))
        clf = SklearnClassifier(classifiers[classifier]).train(trainset)
        # print('%s classifier...completed!' % (classifier))
        accuracy_, precision_, recall_, f1score_ = __scores(clf, testset)
        info['scores'][classifier] = dict()
        info['scores'][classifier]['accuracy'] = accuracy_
        info['scores'][classifier]['precision'] = precision_
        info['scores'][classifier]['recall'] = recall_
        info['scores'][classifier]['f1score'] = f1score_

        if save:
            if not os.path.exists('../models/'):
                os.makedirs('../models/')
            with open('../models/%s.pickle' % (classifier), 'wb') as f:
                pickle.dump(clf, f)
    return info

def predict(samples, classifier):
    """It predicts the input's samples.

    Args:
        samples(list): list of tokenized strings.

    Returns:
        list(): predicted labels.
    """
    samples = samples.apply(FreqDist)
    with open(classifier, 'rb') as f:
        cls = pickle.load(f)
    return cls.classify_many(samples)
