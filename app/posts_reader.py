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


def __remove_urls(self, post):
    """It removes URLs from a string.

    Args:
        post(str): the target post.

    Returns:
        str: the target text without URL.
    """
    search_key = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(search_key, '', post)

def __remove_usernames(self, post):
    """It deletes usernames from a post.

    Args:
        post(str): the target post.

    Returns:
        str: the target post without any username.
    """
    search_key = '@([A-Za-z0-9_]+)'
    return re.sub(search_key, '', post)

def __remove_emojies(self, post_tokenized):
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

def preprocess(self, samples):
    """It pre-processes data.

    Args:
        samples(list): set of samples.

    Returns:
        list: cleaned and tokenized samples. It is the result of the
            preprocessing phase.
    """
    posts_tokenized = list()
    for post in samples:
        # normalizing text in lowercase
        post = post.lower()
        # removing URLs
        post = self.__remove_urls(post)
        # removing usernames
        post = self.__remove_usernames(post)
        # tokenizing text
        post_tokenized = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(post)
        # removing twitter stop words
        post_tokenized = filter(lambda x: x not in ['rt'], post_tokenized)
        # removing punctuations
        post_tokenized = filter(lambda x: x not in punctuation, post_tokenized)
        # removing digits
        post_tokenized = filter(lambda x: x not in digits, post_tokenized)
        # removing emojies
        post_tokenized = self.__remove_emojies(post_tokenized)
        posts_tokenized.append(post_tokenized)
    return posts_tokenized

def postprocess(self, samples, classes):
    """It post-processes data.

    Args:
        samples(list): tokenized strings;
        classes(list): predicted classes.

    Returns:
        list: tuples with the sample, class pair.
    """
    samples = list(map(lambda x: ' '.join(x), samples))
    return list(zip(samples, classes))
