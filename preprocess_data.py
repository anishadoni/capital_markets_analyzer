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

def load_word_vectors(filename):
    # data_dir = os.path.join(BASE_PATH, "data//word2vec//")
    data_dir = BASE_PATH/"data"/"glove_wordvec"
    # filename = os.path.join(data_dir, filename)
    filename = data_dir/filename
    print('-------- loading pre-trained word vector matrix ----------')
    raw_wordvec_file = open(str(filename), encoding = "utf8")
    word_index = []
    raw_vectors = []
    for line in raw_wordvec_file:
        word_index.append(line.split()[0])
        raw_vectors.append(map(float, line.split()[1:]))
    print('Pre-trained word vector matrix loaded.')
    wordvec_df = pd.DataFrame(raw_vectors, word_index)
    raw_wordvec_file.close()
    return wordvec_df

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


def format_movie_reviews(data_path, batch_size=100, use_generator=True):
    """
    This function 'cleans' the imdb movie review dataset and stores it as
    a single .h5 file.
    NOTE
    This function as written relies on hardcoding the number of reviewsself.
    Remember to fix in the future, maybe by adding a .txt file outlining the
    relevant format of the data, like number of classes, number of training
    and testing samples, etc.
    """
    # TODO: Add functionality to automatically calculate shape of dataset.

    def to_wordvec(string):
    	"""
    	Pure function that returns the word vector embedding of a single word.
    	"""
        try:
            return wordvec_df.loc[s.index.intersection(string)].as_matrix()
        except:
            return (np.random.rand(WORDVEC_LENGTH)) #returns vector for unknown words

    wordvec_df = load_word_vectors(WORD_VEC_FILE)
    wordvec_length = wordvec_df.shape[-1]

    os.chdir(str(data_path))
    if use_generator:
        labels = np.zeros(NUM_REVIEWS)
        labels[:(NUM_REVIEWS//2)] = 1

        with h5py.File("review_data.h5") as f:
            dst = f.create_dataset("review_x", shape=(NUM_REVIEWS, 2470, WORDVEC_LENGTH))
            f.create_dataset("review_y", data=labels)

            for idx, text_chunk in enumerate(generate_raw_reviews(data_directory, batch_size)):
                wordvec_matrix = pd.DataFrame(text_chunk).fillna(0)
                saved_matrix = np.zeros((wordvec_matrix.shape[0], wordvec_matrix.shape[1], WORDVEC_LENGTH))

                if saved_matrix.shape[1] < 2470:
                    saved_matrix = np.concatenate((saved_matrix, np.zeros((saved_matrix.shape[0], 2470-saved_matrix.shape[1], saved_matrix.shape[2]))), axis=1)
                    for i in range(wordvec_matrix.shape[0]):
                        output = np.transpose(wordvec_matrix.apply(to_wordvec, axis = 0).values)
                        saved_matrix[i][:output.shape[0]] = output
                        # wordvec_matrix[i] = wordvec_matrix.apply(to_wordvec, axis=1)

                        #add automation to detect shape of input data required
                dst[idx*batch_size:(idx+1)*batch_size] = saved_matrix
        return None
                    # wordvec_matrix.to_hdf("review_data.h5", key="review_x", append=True, compression=2, mode="a")

    text_raw, labels = load_imdb(data_directory)

    # properly formats text from [[train],[test]] to [all_data]
    text_formatted = reduce(lambda x, y: x + y, text_raw, [])
    # creates a dataframe representation of the text, and formats it properly
    wordvec_matrix = pd.DataFrame(text_formatted).fillna('0')
    print(wordvec_matrix.shape)

    # loops through the columns of the text matrix and replaces each word with it's respective word vector embedding
    for i in range(wordvec_matrix.shape[1]):
        wordvec_matrix[i] = wordvec_matrix.apply(to_wordvec, axis = 1)
    os.chdir(str(BASE_PATH/"data"/data_directory))
    wordvec_matrix.to_hdf("review_data.h5", key="review_x", mode="w")
    with h5py.File("review_data.h5", "w") as f:
        f.create_dataset("review_y", data=np.array(labels), dtype="float32")
    os.chdir("../../")

# helper function for removing special characters from strings
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def load_imdb(data_path):
    # NOTE: ADD CHECK FOR DATA FILES
    pos_files = [f for f in data_path.glob("*pos.txt")]
    neg_files = [f for f in data_path.glob("*neg.txt")]
    pos_reviews = []
    for file in pos_files:
        pos_reviews.append([list(map(cleanSentences, line.split())) for line in open(str(file), "r", encoding='utf-8') if len(line.split()) <= MAX_SEQ_LENGTH])
    print("Positive reviews read into memory!")

    neg_reviews = []
    for file in neg_files:
        neg_reviews.append([list(map(cleanSentences, line.split())) for line in open(str(file), "r", encoding='utf-8') if len(line.split()) <= MAX_SEQ_LENGTH])
    print("Negative reviews read into memory!")

    num_reviews = len(pos_reviews[0]) + len(pos_reviews[1]) + len(neg_reviews[0]) + len(neg_reviews[1])
    print("The total number of reviews: ", num_reviews)

    # NOTE the number of reviews here is hardcoded, remember to fix later
    labels = np.zeros(NUM_REVIEWS)
    labels[:len(pos_reviews)] = 1
    labels[len(neg_reviews):] = 0


    return pos_reviews + neg_reviews, labels

# NOTE WORK IN PROGRESS
def generate_raw_reviews(data_directory, batch_size, flag):
    data_dir = BASE_PATH/"data"/data_directory
    pos_files = [f for f in data_dir.glob(flag + "pos.txt")]
    neg_files = [f for f in data_dir.glob(flag + "neg.txt")]

    all_files = pos_files + neg_files
    num_samples_file = sum(1 for line in open(all_files[0]))
    for f in all_files:
        with open(str(f), "r", encoding='utf-8') as file:
            while True:
                text_chunk = [list(map(cleanSentences, line.split())) for line in list(itertools.islice(file, batch_size))]
                if not text_chunk:
                    break
                yield text_chunk

def create_val_raw_reviews(data_path):
	pos_files = [data_path/"train_pos.txt"]
	neg_file = [data_path/"train_neg.txt"]
	num_samples_file = sum(1 for line in open(pos_files[0]))
	length_val = (int) 0.3 * num_samples_file
	g = random.randint(0, num_samples_file - length_val)
	for f in pos_files:
		with open(str(f), "r", encoding='utf-8') as file:
			val_lines = list(itertools.islice(file, start = g, stop = g+length_val))
		with open('val_pos.txt', 'w', encoding = 'utf-8') as val_file:
			for line in val_lines:
				val_file.write(line)
	for f in neg_files:
		with open(str(f), "r", encoding='utf-8') as file:
			val_lines = list(itertools.islice(file, start = g, stop = g+length_val))
		with open('val_neg.txt', 'w', encoding = 'utf-8') as val_file:
			for line in val_lines:
				val_file.write(line)

def make_wordvec_matrix(text, wordvec_file=WORD_VEC_FILE, max_seq_length=MAX_SEQ_LENGTH):
    wordvec_df = load_word_vectors(WORD_VEC_FILE)
    wordvec_length = wordvec_df.shape[-1]

    # properly formats text from [[train],[test]] to [all_data]
    text_formatted = reduce(lambda x, y: x + y, text, [])
    # creates a dataframe representation of the text, and formats it properly
    wordvec_matrix = pd.DataFrame(text_formatted).fillna('0')
    print(wordvec_matrix.shape)

    # pure function that returns the wordvec representation of a single word
    def to_wordvec(string):
        try:
            return wordvec_df.loc[s.index.intersection(string)].tolist()
        except:
            return list(np.random.rand(WORDVEC_LENGTH)) #returns vector for unknown words

    # loops through the columns of the text matrix and replaces each word with it's respective word vector embedding
    for i in range(wordvec_matrix.shape[1]):
        wordvec_matrix[i] = wordvec_matrix.apply(to_wordvec, axis = 1)

    print("wordvec_matrix created for input data")

    return wordvec_matrix

def get_split_data(data, labels, train_split, test_split, cv_split):
    """
    Helper function for splitting data into training, cross_validation, and test.
    Data splits must be given in decimal form, e.g. train_split = 0.8, test_split = 0.1, cv_split = 0.1

    split_data(data, 0.7, 0.2, 0.1) would return 70% train, 20% test, 10% cross validation
    """
    assert train_split + test_split + cv_split <= 1.0, "Total split cannot include more than 100% of trainig data."
    num_samples = np.shape(data)[0]
    shuffle_in_unison(data, labels)
    splitpoint_a = math.floor(train_split*num_samples)
    splitpoint_b = math.floor((train_split + test_split)*num_samples)

    x_train, y_train = data[0:splitpoint_a], labels[0:splitpoint_a]
    x_test, y_test = data[splitpoint_a:splitpoint_b], labels[splitpoint_a:splitpoint_b]
    x_cv, y_cv = data[splitpoint_b:], labels[splitpoint_b:]

    return (x_train, y_train, x_test, y_test, x_cv, y_cv)
