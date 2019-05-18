"""

.. _Google Python Style Guide
    https://github.com/google/styleguide/blob/gh-pages/pyguide.md
"""

__copyright__ = 'Copyright 2019, University of Messina'
__author__ = 'Lorenzo Carnevale <lorenzocarnevale@gmail.com>
__credits__ = ''
__description__ = ''

import collections
from nltk.metrics import scores

def precision_recall(classifier, testfeats):
	"""
	"""
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)

	for i, (feats, label) in enumerate(testfeats):
		refsets[label].add(i)
		observed = classifier.classify(feats)
		testsets[observed].add(i)

	precisions = {}
	recalls = {}

	for label in classifier.labels():
		precisions[label] = scores.precision(refsets[label], testsets[label])
		recalls[label] = scores.recall(refsets[label], testsets[label])

	return precisions, recalls

def f1score(recalls, precisions):
	"""
	"""
	f1scores = dict()
	for key in recalls.keys():
		if recalls[key] == None or precisions[key] == None:
			f1score_ = None
		else:
			f1score_ = (2 * (recalls[key] * precisions[key])) / (recalls[key] + precisions[key])
		f1scores[key] = f1score_
	return f1scores
