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
import os
import csv
# local libraries
from models.metrics import precision_recall
from models.metrics import f1score
# third parties libraries
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


class Classifier:
    """
    """
    def __init__(self, posts_tokenized, tests):
        """

        Args:
            max_ngrams(list):
        """
        self.posts_tokenized = posts_tokenized
        self.tests = tests

        self.vocabulary_sizes = list()

        self.__settings = [
            {
                "classifier": BernoulliNB(),
                "scores": {
                    "accuracy": list(),
                    "precision": list(),
                    "recall": list(),
                    "f1score": list()
                }
            },
            {
                "classifier": MultinomialNB(),
                "scores": {
                    "accuracy": list(),
                    "precision": list(),
                    "recall": list(),
                    "f1score": list()
                }
            },
            {
                "classifier": LogisticRegression(),
                "scores": {
                    "accuracy": list(),
                    "precision": list(),
                    "recall": list(),
                    "f1score": list()
                }
            },
            {
                "classifier": SGDClassifier(),
                "scores": {
                    "accuracy": list(),
                    "precision": list(),
                    "recall": list(),
                    "f1score": list()
                }
            },
            {
                "classifier": SVC(),
                "scores": {
                    "accuracy": list(),
                    "precision": list(),
                    "recall": list(),
                    "f1score": list()
                }
            },
            {
                "classifier": LinearSVC(),
                "scores": {
                    "accuracy": list(),
                    "precision": list(),
                    "recall": list(),
                    "f1score": list()
                }
            }
        ]

    def __sentiment(self, n, dataset):
        """
        """
        for post_tokenized in self.posts_tokenized:
            for ngram in ngrams(post_tokenized, n):
                blob = TextBlob(" ".join(ngram))
                polarity = blob.sentences[0].sentiment.polarity
                if polarity <= -0.3:
                    ngram_class = "alarm"
                elif polarity >= 0.3:
                    ngram_class = "notalarm"
                else:
                    ngram_class = "suspect"
                dataset.append((FreqDist(ngram), ngram_class))
        self.vocabulary_sizes.append(len(dataset))

        return dataset

    def __scores(self, clf, testset):
        """
        """
        accuracy_ = accuracy(clf, testset)
        precision_, recall_ = precision_recall(clf, testset)
        f1score_ = f1score(precision_, recall_)
        return accuracy_, precision_, recall_, f1score_

    def run(self):
        """
        """
        prev_dataset = list()
        curr_dataset = list()
        for n in self.tests:
            print("START ANALYSIS with ngrams test number: %s" % (n))

            curr_dataset = list()
            curr_dataset = self.__sentiment(n, curr_dataset)
            dataset =  prev_dataset + curr_dataset

            trainset, testset = train_test_split(dataset, test_size=0.33, random_state=42)
            print('\tVocabulary size %s, trainset %s, testset %s' % (len(dataset),len(trainset),len(testset)))

            for setting in self.__settings:
                classifier = setting['classifier']
                clf = SklearnClassifier(classifier).train(trainset)
                accuracy_, precision_, recall_, f1score_ = self.__scores(clf, testset)
                setting['scores']['accuracy'].append(accuracy_)
                setting['scores']['precision'].append(precision_)
                setting['scores']['recall'].append(recall_)
                setting['scores']['f1score'].append(f1score_)

            prev_dataset = curr_dataset

    def export(self):
        """
        """
        if not os.path.exists('../results'):
            os.makedirs('../results')
        # save on csv
        with open('../results/accuracy_incr2step.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.tests)
            # writer.writerow(self.vocabulary_sizes)
            for setting in self.__settings:
                writer.writerow(setting['scores']['accuracy'])

        with open('../results/precision_incr2step.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.tests)
            for setting in self.__settings:
                row = [ score for elem in setting['scores']['precision'] for score in elem.values() ]
                writer.writerow(row)

        with open('../results/recall_incr2step.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.tests)
            for setting in self.__settings:
                row = [ score for elem in setting['scores']['recall'] for score in elem.values() ]
                writer.writerow(row)

        with open('../results/f1score_incr2step.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.tests)
            for setting in self.__settings:
                row = [ score for elem in setting['scores']['f1score'] for score in elem.values() ]
                writer.writerow(row)
