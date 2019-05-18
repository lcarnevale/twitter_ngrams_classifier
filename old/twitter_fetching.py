#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	This application fetches tweets from Twitter.

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


import sys
import json
import tweepy


#==============================
# ENV SETTINGS
#==============================
conf = json.load(open(sys.argv[1])) # load configuration file
CONSUMER_KEY = conf["consumer_key"]
CONSUMER_SECRET = conf["consumer_secret"]
ACCESS_KEY = conf["access_key"]
ACCESS_SECRET = conf["access_secret"]
HASHTAG = '#'+sys.argv[2]
OUT_DIRECTORY = '../datasets/'
filename = OUT_DIRECTORY+"dataset-"+HASHTAG+".txt"



def main():

	#==============================
	# TWITTER SETTINGS
	#==============================
	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
	api = tweepy.API(auth, wait_on_rate_limit=True) # authenticate against the Twitter API

	#==============================
	# FETCHING
	#==============================
	try:
		posts = list()
		print 'Start fetching...'
		for tweet in tweepy.Cursor(api.search,q=HASHTAG,lang="en",
				rpp=100,tweet_mode='extended').items():
			posts.append(tweet.full_text)
	except KeyboardInterrupt:
		print "\nforced closing"
	finally:
		posts = list(set(posts))
		filename = OUT_DIRECTORY+"dataset-"+HASHTAG+".json"
		with open(filename, 'w') as outfile:
			json.dump(posts, outfile)
		print "saved on ", filename


if __name__ == '__main__':
	main()
	exit()
