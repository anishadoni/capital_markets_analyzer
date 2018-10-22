import os
import numpy as np
import csv
import pandas as pd
import tweepy
import jsonpickle
import json
from tweepy.streaming import StreamListener
from tweepy import Stream
import xml.etree.ElementTree as et
import stock_crawler
import preprocess_data

# global variables
BASE_PATH = os.path.dirname(os.path.realpath(__file__))

# data is returned in the form of a pandas DataFrame
def get_twitter_api_keys(api_key_file):
	base_path = os.path.dirname(os.path.realpath(__file__))
	csv_file = os.path.join(base_path, api_key_file)

	_api_keys = pd.read_csv(csv_file, header = None, index_col = 0).T
	return _api_keys

# pretty self explanatory lol 
####### NOTE ####### edit to make more useful, like making it list out timeline tweets from specified user
def test_tweepy_api(api_key_file):
	api = get_twitter_api_handle(api_key_file)


	api.search(q="hello")
	print("Twitter api access successful!")


# returns twitter api handle using tweepy
######## NOTE ##### edit to access api keys from a text file
# also add functionality to use multiple api access keys -> effectively return a list of tweepy APIs
def get_twitter_api_handle(api_key_file, APP_AUTH = True):
	api_keys = get_twitter_api_keys(api_key_file)

	_consumer_key = "nWpuKXeX0cuRZnqjB90ToGZpm"
	_consumer_secret = "79WnWiS76R3IeVrawAdzMIv74JlqyMIO2cr975qJqJzMdehWbF"

	
	auth = tweepy.AppAuthHandler(_consumer_key, _consumer_secret)
	
	# auth.set_access_token(_access_token, _access_token_secret)	

	api = tweepy.API(auth)

	return api

def download_tweets(filename, api_key_file):
	maxTweets = 1000
	tweetsPerQry = 100

	base_path = os.path.dirname(os.path.realpath(__file__))

	company_list = preprocess_data.load_company_frame(filename = filename)


	api = get_twitter_api_handle(api_key_file)

	# new implementation for iterating through tweets using tweepy's Cursor api
	for company in company_list.itertuples():
		ticker = company[1]
		query = company[2]
		
		print("--------- downloading tweets about " + query + " -----------")

		_tweets = tweepy.Cursor(api.search, q = query, count = tweetsPerQry, lang = "en").items(maxTweets)

		print("Successfully downloaded " + str(maxTweets) + " tweets about " + query + ".\n")

		saveTweets(ticker = ticker, tweetData = _tweets, companyName = query)


# saves an array of tweets 
def saveTweets(ticker, tweetData, companyName):
	data_dir = os.path.join(BASE_PATH, "data\\" + ticker)

	print("-------- saving tweets about " + companyName + " ---------")

	tweetCount = 0

	with open(data_dir + "\\" + ticker + "_twitter_data.json", 'w') as json_data_file:
		for tweet in tweetData:
			json_data_file.write(jsonpickle.encode(tweet._json, unpicklable = False) + '\n')
			tweetCount+=1
		print("Successfully saved " + str(tweetCount) + " tweets about " + companyName + ".\n")


if __name__ == "__main__":
	download_tweets("company_stock.xml", "twitter_api_keys.csv")
	




