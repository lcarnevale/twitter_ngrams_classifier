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
import re
from string import digits
from string import punctuation
# third parties libraries
from nltk.tokenize import TweetTokenizer


def __remove_urls(post):
    """It removes URLs from a string.

    Args:
        post(str): the target post.

    Returns:
        str: the target text without URL.
    """
    search_key = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(search_key, '', post)

def __remove_usernames(post):
    """It deletes usernames from a post.

    Args:
        post(str): the target post.

    Returns:
        str: the target post without any username.
    """
    search_key = '@([A-Za-z0-9_]+)'
    return re.sub(search_key, '', post)

def __remove_emojies(post_tokenized):
    """It deletes emojies from a post.

    Args:
        tokens(list): the tokenized target post.

    Returns:
        list: the tokenized target post without any emoji.
    """
    new_post_tokenized = []
    for word in post_tokenized:
        try:
            str(word)
            new_post_tokenized.append(word)
        except UnicodeEncodeError:
            pass
    return new_post_tokenized

def preprocess(samples):
    """It pre-processes data.

    Args:
        samples(obj:'pandas.core.series.Series'): set of samples.

    Returns:
        obj:'pandas.core.series.Series': cleaned and tokenized samples. It
            is the result of the preprocessing phase.
    """
    def process(sample):
        # normalizing text in lowercase
        sample = sample.lower()
        # removing URLs
        sample = __remove_urls(sample)
        # removing usernames
        sample = __remove_usernames(sample)
        # tokenizing text
        sample_tokenized = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(sample)
        # removing twitter stop words
        sample_tokenized = filter(lambda x: x not in ['rt'], sample_tokenized)
        # removing punctuations
        sample_tokenized = filter(lambda x: x not in punctuation, sample_tokenized)
        # removing digits
        sample_tokenized = filter(lambda x: x not in digits, sample_tokenized)
        # removing emojies
        sample_tokenized = __remove_emojies(sample_tokenized)
        return sample_tokenized
    return samples.apply(lambda x: process(x))

def postprocess(samples, classes):
    """Post-processes data.

    Args:
        samples (list): target samples;
        classes (list): predicted classes.

    Returns:
        list: tuples with the sample, class pair.
    """
    return list(zip(samples, classes))
