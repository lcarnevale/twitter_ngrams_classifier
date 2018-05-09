#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	TODO

	The scope of the application is academic research.
	It is part of the NLP activities carried out at University
	of Messina.
"""

__author__ = "Lorenzo Carnevale"
__copyright__ = None
__credits__ = ["Lorenzo Carnevale"]
__license__ = None
__version__ = "1"
__maintainer__ = "Lorenzo Carnevale"
__email__ = "lcarnevale@unime.it"
__status__ = "Prototype"
__doi__ = ['10.5220/0006750606800687']

"""
	@conference{ai4health18,
		author={Giacomo Fiumara and Antonio Celesti and Antonino Galletta and Lorenzo Carnevale and Massimo Villari},
		title={Applying Artificial Intelligence in Healthcare Social Networks to Identity Critical Issues in Patients’ Posts},
		booktitle={Proceedings of the 11th International Joint Conference on Biomedical Engineering Systems and Technologies - Volume 5: AI4Health,},
		year={2018},
		pages={680-687},
		publisher={SciTePress},
		organization={INSTICC},
		doi={10.5220/0006750606800687},
		isbn={978-989-758-281-3},
	}
"""

import sys
import json
import re
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from string import digits
from textblob import TextBlob
from nltk import ngrams
from nltk import FreqDist
from nltk.tokenize import TweetTokenizer
from nltk.classify import SklearnClassifier
from nltk.classify import accuracy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from scipy.interpolate import spline
from apis.tweet import Preprocess


#==========================================
# DATA ANALYSIS SETTINGS
#==========================================
hashtag = re.search('dataset-#(.*).json',sys.argv[1]).group(1)
posts = json.load(open(sys.argv[1])) # load dataset from argv
tests = [1,2,3,4,5,6,7,8] # ngram number for testing
pp = Preprocess()
print "\tHashtag: %s" % (hashtag)
print "\tPosts number: %s" % (len(posts))
vocabulary_sizes = list()
BernoulliNB_acc = list()
MultinomialNB_acc = list()
LogisticRegression_acc = list()
SGDClassifier_acc = list()
SVC_acc = list()
LinearSVC_acc = list()


#==========================================
# PREPROCESSING
#==========================================
posts_tokenized = list()
for post in posts:
	post = post.lower()
	post = pp.remove_urls(post)
	post = pp.remove_usernames(post)
	post = pp.remove_hashtag(post)
	post_tokenized = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(post)
	post_tokenized = filter(lambda x: x not in ['rt'], post_tokenized)
	post_tokenized = filter(lambda x: x not in punctuation, post_tokenized)
	post_tokenized = filter(lambda x: x not in digits, post_tokenized)
	post_tokenized = pp.remove_emojies(post_tokenized)
	posts_tokenized.append(post_tokenized)

prev_dataset = list()
curr_dataset = list()
for test in tests:
	n = test # select the ngram number.
	print "START ANALYSIS with ngrams test number: %s" % (n)

	curr_dataset = list()

	#==========================================
	# DICTIONARY CREATION: SENTIMENT ANALYSIS
	#==========================================
	for post_tokenized in posts_tokenized:
		for ngram in ngrams(post_tokenized, n):
			blob = TextBlob(" ".join(ngram))
			polarity = blob.sentences[0].sentiment.polarity
			if polarity <= -0.3:
				ngram_class = "alarm"
			elif polarity >= 0.3:
				ngram_class = "notalarm"
			else:
				ngram_class = "suspect"
			curr_dataset.append((FreqDist(ngram), ngram_class))
	dataset =  prev_dataset + curr_dataset
	vocabulary_sizes.append(len(dataset))


	#==========================================
	# TRAIN-TEST
	#==========================================
	trainset, testset = train_test_split(dataset, test_size=0.33, random_state=42)
	print '\tVocabulary size %s, trainset %s, testset %s' % (len(dataset),len(trainset),len(testset))

	#==========================================
	# CLASSIFICATION: NAIVE BAYES
	#==========================================
	clf = SklearnClassifier(BernoulliNB()).train(trainset)
	score = accuracy(clf, testset)
	BernoulliNB_acc.append(score)
	print "\tBernoulliNB with %s ngrams - score %s" % (n, score)

	clf = SklearnClassifier(MultinomialNB()).train(trainset)
	score = accuracy(clf, testset)
	MultinomialNB_acc.append(score)
	print "\tMultinomialNB with %s ngrams - score %s" % (n, score)

	#==========================================
	# CLASSIFICATION: LINEAR MODEL
	#==========================================
	clf = SklearnClassifier(LogisticRegression()).train(trainset)
	score = accuracy(clf, testset)
	LogisticRegression_acc.append(score)
	print("\tLogisticRegression with %s ngrams - score %s") % (n, score)

	clf = SklearnClassifier(SGDClassifier()).train(trainset)
	score = accuracy(clf, testset)
	SGDClassifier_acc.append(score)
	print("\tSGDClassifier with %s ngrams - score %s") % (n, score)

	#==========================================
	# CLASSIFICATION: SUPPORT VECTOR MACHINE
	#==========================================
	clf = SklearnClassifier(SVC()).train(trainset)
	score = accuracy(clf, testset)
	SVC_acc.append(score)
	print("\tSVC with %s ngrams - score %s") % (n, score)

	clf = SklearnClassifier(LinearSVC()).train(trainset)
	score = accuracy(clf, testset)
	LinearSVC_acc.append(score)
	print("\tLinearSVC with %s ngrams - score %s") % (n, score)

	prev_dataset = curr_dataset

# save on csv
ofile = open('../results/result-incr2step_acc.csv', 'wb')
writer = csv.writer(ofile, quoting=csv.QUOTE_ALL)
writer.writerow(tests)
writer.writerow(vocabulary_sizes)
writer.writerow(BernoulliNB_acc)
writer.writerow(MultinomialNB_acc)
writer.writerow(LogisticRegression_acc)
writer.writerow(SGDClassifier_acc)
writer.writerow(SVC_acc)
writer.writerow(LinearSVC_acc)
ofile.close()

# interpolation for smoother lines
tests = np.array(tests)
xnew = np.linspace(tests.min(), tests.max(), 300) #300 represents number of points to make between T.min and T.max

# plot vocabulary sizes
fig, ax = plt.subplots()
vocabulary_sizes = spline(tests,vocabulary_sizes,xnew)
ax.plot(xnew, vocabulary_sizes)
ax.set(xlabel='Ngrams', ylabel='Size',
       title='Vocabulary size')
ax.grid()
fig.savefig("../results/vocabulary-incr2step_size.png")

# plot naive bayes accuracy
fig, ax = plt.subplots()
BernoulliNB_acc = spline(tests,BernoulliNB_acc,xnew)
MultinomialNB_acc = spline(tests,MultinomialNB_acc,xnew)
ax.plot(xnew, BernoulliNB_acc, label="BernoulliNB")
ax.plot(xnew, MultinomialNB_acc, label="MultinomialNB")
ax.set(xlabel='Ngrams', ylabel='Accuracy',
       title='Naive Bayes algorithms accuracy')
ax.legend(loc='lower left')
ax.grid()
fig.savefig("../results/naivebayes-incr2step_acc.png")

# plot linear model accuracy
fig, ax = plt.subplots()
LogisticRegression_acc = spline(tests,LogisticRegression_acc,xnew)
SGDClassifier_acc = spline(tests,SGDClassifier_acc,xnew)
ax.plot(xnew, LogisticRegression_acc, label="LogisticRegression")
ax.plot(xnew, SGDClassifier_acc, label="SGDClassifier")
ax.set(xlabel='Ngrams', ylabel='Accuracy',
       title='Linear model algorithms accuracy')
ax.legend(loc='lower left')
ax.grid()
fig.savefig("../results/linearmodel-incr2step_acc.png")

# plot support vector machine accuracy
fig, ax = plt.subplots()
SVC_acc = spline(tests,SVC_acc,xnew)
LinearSVC_acc = spline(tests,LinearSVC_acc,xnew)
ax.plot(xnew, SVC_acc, label="SVC")
ax.plot(xnew, LinearSVC_acc, label="LinearSVC")
ax.set(xlabel='Ngrams', ylabel='Accuracy',
       title='Support Vector Machine algorithms accuracy')
ax.legend(loc='lower left')
ax.grid()
fig.savefig("../results/svm-incr2step_acc.png")

exit()