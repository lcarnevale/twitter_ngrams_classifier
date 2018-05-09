#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	This application gathers any useful tools to preprocess
	tweets.

	The scope of the application is academic research.
	It is part of the NLP activities carried out at University
	of Messina.
"""

__author__ = "Lorenzo Carnevale"
__copyright__ = None
__credits__ = ["Lorenzo Carnevale"]
__license__ = None
__version__ = "0.1"
__maintainer__ = "Lorenzo Carnevale"
__email__ = "lcarnevale@unime.it"
__status__ = "Prototype"


import re


class Preprocess():
	"""
		Provide methods for cleaning tweets.
	"""

	def __init__(self):
		"""
			The init method is empty.
		"""
		pass


	def remove_urls(self, post):
		"""
			This method removes urls from a post.

			Args:
				post: the tweet.

			Returns:
				The post without any url.
		"""

		search_key = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
		return re.sub(search_key, '', post)

	def remove_usernames(self, post):
		"""
			This method deletes usernames from a tweet.

			Args:
				post: the tweet.

			Returns
				The tweet without any username.
		"""

		search_key = '@([A-Za-z0-9_]+)'
		return re.sub(search_key, '', post)

	def remove_hashtag(self, post):
		"""
			This method deletes hashtags from a tweet.

			Args:
				post: the tweet.

			Returns:
				The tweet without any hashtag.
		"""

		search_key = '#([A-Za-z0-9_]+)'
		return re.sub(search_key, '', post)

	def remove_emojies(self, post_tokenized):
		"""
			This method deletes emojies from a tweet.

			Argv:
				post_tokenized: the tweet tokenized, i.e. a list of words.

			Returns:
				The tweet tokenized without any emoji.
		"""

		new_post_tokenized = []
		for word in post_tokenized:
			try:
				str(word)
				new_post_tokenized.append(word)
			except UnicodeEncodeError:
				pass
		return new_post_tokenized