import pandas as pd
import numpy as np
import h5py
import os
import math
from os.path import isfile, join, dirname
from os import listdir
from pathlib import Path
import json
import glob
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import reduce
from generators import *
import itertools
import random
import re

# BASE_PATH = dirname(os.path.realpath(__file__))
BASE_PATH = Path.cwd()
NUM_WORD_EMBEDDINGS = 400000 # number of word embeddings used
MAX_SEQ_LENGTH_SHORT = 50
MAX_SEQ_LENGTH = 2500
WORDVEC_LENGTH = 50
NUM_CLASSES = 2
NUM_PROCESSES = 4 # ideally should be set to number of cores on cpu
NUM_REVIEWS = 50000
FILE_LENGTHS = {"train": 17500, "test": 25000, "val": 7500}
WORD_VEC_FILE = "glove.6B.50d.txt"

def load_company_frame(filename):

    xml_file = BASE_PATH/"data"/"twitter"/filename

    tree = et.parse(str(xml_file))

    root = tree.getroot()

    # view source .xml file for data storage format
    col_names = ['TICKER', 'HASH']
    index = np.linspace(1, len(root), num = 13)
    company_df = pd.DataFrame(index = index, columns = col_names)

    iterator = 1
    for child in root:
        company_df.loc[iterator]['TICKER'] = child[2].text
        company_df.loc[iterator]['HASH'] = child[1].text
        iterator = iterator + 1

    return company_df

def load_tickers(filename):
    _tickers = []
    # _companies = []
    xml_file = BASE_PATH/"data"/"twitter"/filename

    tree = et.parse(str(xml_file))

    root = tree.getroot()

    # review company_stock.xml for data organization format
    for child in root:
        # _companies.append(child[0].text)
        _tickers.append(child[2].text)
    # new_product = et.SubElement(root, "company", attrib={"id": "9"})
    return _tickers

# loads company hash values
def load_company_hash(filename):
    _company_hash_list = []
    xml_file = BASE_PATH/"data"/"twitter"/filename
    tree = et.parse(str(xml_file))
    root = tree.getroot()
    # check company_stock.xml in BASE_PATH//data to determine data storage format
    for child in root:
        _company_hash_list.append(child[1].text)

    return _company_hash_list

def load_tweets(filename):
    # tweetsPerFile = 1000
    company_list = load_company_frame(filename = filename)
    all_tweets = []
    # col_names = load_tickers(filename = filename)
    # index = np.linspace(1, 1000, num = 1000)

    # all_tweets_df = pd.DataFrame(index = index, columns = col_names)
    for company in company_list.itertuples():
        ticker = company[1]
        name = company[2]

        tweets = []
        # data_dir = os.path.join(BASE_PATH, "data//twitter//" + ticker)
        data_dir = BASE_PATH/"data"/"twitter"/ticker

        print("---------- loading twitter data for " + name + " ----------")

        for line in open(data_dir/"twitter_data.json", 'r'):
            raw_tweet = json.loads(line)
            tweets.append(raw_tweet['text'])
            # tweets.append(json.loads(line))

        all_tweets.append(tweets)

    return all_tweets

def shuffle_in_unison(a,b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def load_short_movie_reviews():
    data_dir = join(BASE_PATH, "data\\short_reviews\\")
    pos_reviews_file = "positive.txt"
    neg_reviews_file = "negative.txt"

    pos_reviews_raw = [line.split() for line in open(join(data_dir, pos_reviews_file), "r")]
    print("Positive reviews loaded.")
    neg_reviews_raw = [line.split() for line in open(join(data_dir, neg_reviews_file), "r")]
    print("Negative reviews loaded.")
    num_reviews = len(pos_reviews_raw + neg_reviews_raw)

    labels = np.zeros((num_reviews, NUM_CLASSES))
    labels[:len(pos_reviews_raw)] = [1,0]
    labels[len(pos_reviews_raw):] = [0,1]

    num_words = [len(l) for l in pos_reviews_raw]
    num_words += [len(l) for l in neg_reviews_raw]

    num_samples = len(num_words)
    avg_len = sum(num_words)/num_samples

    print('The total number of words is ', sum(num_words))
    print('The average number of words per review is ', avg_len)

    # plt.hist(num_words, 50)
    # plt.xlabel('Sequence Length')
    # plt.ylabel('Frequency')
    # plt.axis([0, 80, 0, 900])
    # plt.show()
    return pos_reviews_raw + neg_reviews_raw, labels
