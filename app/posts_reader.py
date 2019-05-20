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

class PostsReader:
    """
    """
    def __init__(self, posts):
        """
        """
        self.posts = posts

    def remove_urls(self, post):
        """It removes URLs from a string.

        Args:
            post(str): the target post.

        Returns:
            str: the target text without URL.
        """
        search_key = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(search_key, '', post)

    def remove_usernames(self, post):
        """It deletes usernames from a post.

        Args:
            post(str): the target post.

        Returns:
            str: the target post without any username.
        """
        search_key = '@([A-Za-z0-9_]+)'
        return re.sub(search_key, '', post)

    def remove_hashtag(self, post):
        """It deletes hashtags from a post.

        Args:
            post(str): the target post.

        Returns:
            str: the target post without any hashtag.
        """
        search_key = '#([A-Za-z0-9_]+)'
        return re.sub(search_key, '', post)

    def remove_emojies(self, post_tokenized):
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

    def info(self):
        """
        """
        return {
            "size": len(posts)
        }

    def preprocess(self):
        """
        """
        posts_tokenized = list()
        for post in self.posts:
            post = post.lower()
            post = self.remove_urls(post)
            post = self.remove_usernames(post)
            post = self.remove_hashtag(post)
            post_tokenized = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(post)
            post_tokenized = filter(lambda x: x not in ['rt'], post_tokenized)
            post_tokenized = filter(lambda x: x not in punctuation, post_tokenized)
            post_tokenized = filter(lambda x: x not in digits, post_tokenized)
            post_tokenized = self.remove_emojies(post_tokenized)
            posts_tokenized.append(post_tokenized)
        return posts_tokenized

    def postprocess(self, samples, classes):
        """It post processes data.

        Args:
            samples(list): tokenized strings;
            classes(list): predicted classes.

        Returns:
            list: tuples with the sample, class pair.
        """
        samples = list(map(lambda x: ' '.join(x), samples))
        return list(zip(samples, classes))
