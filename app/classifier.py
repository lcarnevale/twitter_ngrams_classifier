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


vocabulary_sizes = list()

__settings = [
    {
        "name": "BernoulliNB",
        "classifier": BernoulliNB(),
        "scores": {
            "accuracy": list(),
            "precision": list(),
            "recall": list(),
            "f1score": list()
        }
    },
    {
        "name": "MultinomialNB",
        "classifier": MultinomialNB(),
        "scores": {
            "accuracy": list(),
            "precision": list(),
            "recall": list(),
            "f1score": list()
        }
    },
    {
        "name": "LogisticRegression",
        "classifier": LogisticRegression(),
        "scores": {
            "accuracy": list(),
            "precision": list(),
            "recall": list(),
            "f1score": list()
        }
    },
    {
        "name": "SGDClassifier",
        "classifier": SGDClassifier(),
        "scores": {
            "accuracy": list(),
            "precision": list(),
            "recall": list(),
            "f1score": list()
        }
    },
    {
        "name": "SVC",
        "classifier": SVC(),
        "scores": {
            "accuracy": list(),
            "precision": list(),
            "recall": list(),
            "f1score": list()
        }
    },
    {
        "name": "LinearSVC",
        "classifier": LinearSVC(),
        "scores": {
            "accuracy": list(),
            "precision": list(),
            "recall": list(),
            "f1score": list()
        }
    }
]

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

def train(samples_tokenized, ngram_min, ngram_max):
    """
    """
    dataset = __generate_features(samples_tokenized, ngram_min, ngram_max)
    trainset, testset = train_test_split(dataset, test_size=0.33, random_state=42)
    print('Vocabulary size %s, trainset %s, testset %s' % (len(dataset),len(trainset),len(testset)))

    for setting in __settings:
        classifier = setting['classifier']
        clf = SklearnClassifier(classifier).train(trainset)
        accuracy_, precision_, recall_, f1score_ = __scores(clf, testset)
        setting['scores']['accuracy'].append(accuracy_)
        setting['scores']['precision'].append(precision_)
        setting['scores']['recall'].append(recall_)
        setting['scores']['f1score'].append(f1score_)

        with open('%s.pickle' % (setting['name']), 'wb') as f:
            pickle.dump(clf, f)

def predict(samples):
    """It predicts the input's samples.

    Args:
        samples(list): list of tokenized strings.

    Returns:
        list(): predicted labels.
    """
    with open('MultinomialNB.pickle', 'rb') as f:
        cls = pickle.load(f)
        test = list()
        for sample in samples:
            test.append(FreqDist(sample))

        return cls.classify_many(test)

def export():
    """
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # save on csv
    with open('../results/accuracy_incr.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(self.tests)
        # writer.writerow(self.vocabulary_sizes)
        for setting in self.__settings:
            writer.writerow(setting['scores']['accuracy'])

    with open('../results/precision_incr.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(self.tests)
        for setting in self.__settings:
            row = [ score for elem in setting['scores']['precision'] for score in elem.values() ]
            writer.writerow(row)

    with open('../results/recall_incr.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(self.tests)
        for setting in self.__settings:
            row = [ score for elem in setting['scores']['recall'] for score in elem.values() ]
            writer.writerow(row)

    with open('../results/f1score_incr.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(self.tests)
        for setting in self.__settings:
            row = [ score for elem in setting['scores']['f1score'] for score in elem.values() ]
            writer.writerow(row)
