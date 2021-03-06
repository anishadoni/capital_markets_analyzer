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
from preprocess_data import shuffle_in_unison
import more_itertools as mitools
import random
import re

BASE_PATH = Path.cwd()
NUM_WORD_EMBEDDINGS = 400000  # number of word embeddings used
MAX_SEQ_LENGTH_SHORT = 50
MAX_SEQ_LENGTH = 2500
WORDVEC_LENGTH = 50
NUM_CLASSES = 2
NUM_PROCESSES = 4  # ideally should be set to number of cores on cpu
NUM_REVIEWS = 50000
FILE_LENGTHS = {"train": 17500, "test": 25000, "val": 7500}
WORD_VEC_FILE = "glove.6B.50d.txt"


def load_vector_embeddings(filename):
    # data_dir = os.path.join(BASE_PATH, "data//word2vec//")
    data_dir = BASE_PATH/"data"/"glove_wordvec"
    # filename = os.path.join(data_dir, filename)
    filename = data_dir/filename
    print('-------- loading pre-trained word vector matrix ----------')
    raw_wordvec_file = open(str(filename), encoding="utf8")
    word_index = []
    raw_vectors = []
    for line in raw_wordvec_file:
        word_index.append(line.split()[0])
        raw_vectors.append(map(float, line.split()[1:]))
    print('Pre-trained word vector matrix loaded.')
    wordvec_df = pd.DataFrame(raw_vectors, word_index)
    raw_wordvec_file.close()
    return wordvec_df


def format_movie_reviews(data_path, flag="train", batch_size=100, use_generator=True):
    """
    This function 'cleans' the imdb movie review dataset and stores it as
    a single .h5 file.
    NOTE
    This function as written relies on hardcoding the number of reviews.
    Remember to fix in the future, maybe by adding a .txt file outlining the
    relevant format of the data, like number of classes, number of training
    and testing samples, etc.
    """
    # TODO: Add functionality to automatically calculate shape of dataset.
    def to_wordvec(s):
        try:
            return wordvec_df.loc[s.index.intersection(s)].as_matrix()
        except:
            # returns vector for unknown words
            return (np.random.rand(WORDVEC_LENGTH))

    wordvec_df = load_vector_embeddings(WORD_VEC_FILE)
    wordvec_length = wordvec_df.shape[-1]

    data_dir = BASE_PATH/"data"/data_path
    os.chdir(str(data_dir))
    if use_generator:

        with h5py.File("review_data.h5") as f:
            dst = f.create_dataset(
                "x_" + flag, shape=(NUM_REVIEWS, 2470, WORDVEC_LENGTH))
            labels = f.create_dataset(
                "y_" + flag, shape=(FILE_LENGTHS[flag], 1))

            for idx, text_chunk in enumerate(generate_raw_reviews_json(data_path, "train", batch_size)):
                wordvec_matrix = pd.DataFrame(
                    [t["x"] for t in text_chunk]).fillna(0)
                saved_matrix = np.zeros(
                    (wordvec_matrix.shape[0], wordvec_matrix.shape[1], WORDVEC_LENGTH))
                if saved_matrix.shape[1] < 2470:
                    saved_matrix = np.concatenate((saved_matrix, np.zeros(
                        (saved_matrix.shape[0], 2470-saved_matrix.shape[1], saved_matrix.shape[2]))), axis=1)
                    for i in range(wordvec_matrix.shape[0]):
                        output = np.transpose(
                            wordvec_matrix.apply(to_wordvec, axis=0).values)
                        saved_matrix[i][:output.shape[0]] = output
                        # wordvec_matrix[i] = wordvec_matrix.apply(to_wordvec, axis=1)

                        # add automation to detect shape of input data required
                dst[idx*batch_size:(idx+1)*batch_size] = saved_matrix
                labels[idx*batch_size:(idx+1)*batch_size] = np.expand_dims(
                    np.array([d["y"] for d in text_chunk]), axis=1)
        return None
        # wordvec_matrix.to_hdf("review_data.h5", key="review_x", append=True, compression=2, mode="a")

    text_raw, labels = load_imdb(data_dir)

    # properly formats text from [[train],[test]] to [all_data]
    text_formatted = reduce(lambda x, y: x + y, text_raw, [])
    # creates a dataframe representation of the text, and formats it properly
    wordvec_matrix = pd.DataFrame(text_formatted).fillna('0')
    print(wordvec_matrix.shape)

    # loops through the columns of the text matrix and replaces each word with it's respective word vector embedding
    for i in range(wordvec_matrix.shape[1]):
        wordvec_matrix[i] = wordvec_matrix.apply(to_wordvec, axis=1)
    os.chdir(str(BASE_PATH/"data"/data_dir))
    wordvec_matrix.to_hdf("review_data.h5", key="review_x", mode="w")
    with h5py.File("review_data.h5", "w") as f:
        f.create_dataset("review_y", data=np.array(labels), dtype="float32")
    os.chdir("../../")


def load_imdb(data_path, flag):
    # NOTE: ADD CHECK FOR DATA FILES
    pos_files = [f for f in data_path.glob(flag + "*pos.txt")]
    neg_files = [f for f in data_path.glob(flag + "*neg.txt")]
    pos_reviews = []
    for file in pos_files:
        pos_reviews.append([list(map(cleanSentences, line.split())) for line in open(
            str(file), "r", encoding='utf-8') if len(line.split()) <= MAX_SEQ_LENGTH])
    print("Positive reviews read into memory!")

    neg_reviews = []
    for file in neg_files:
        neg_reviews.append([list(map(cleanSentences, line.split())) for line in open(
            str(file), "r", encoding='utf-8') if len(line.split()) <= MAX_SEQ_LENGTH])
    print("Negative reviews read into memory!")

    num_reviews = len(pos_reviews[0]) + len(pos_reviews[1]) + \
        len(neg_reviews[0]) + len(neg_reviews[1])
    print("The total number of reviews: ", num_reviews)

    # NOTE the number of reviews here is hardcoded, remember to fix later
    labels = np.zeros(NUM_REVIEWS)
    labels[:len(pos_reviews)] = 1
    labels[len(neg_reviews):] = 0

    return pos_reviews + neg_reviews, labels


def reviews_to_json(data_path, r=0.1):
    """
    NOTE: UPDATE TO REMOVE REPEATED CODE -> Possibly create a helper function
    or a dictionary to map all raw .txt files.
    Transfers all reviews in raw format to json files. Also creates a validation
    set from the training set, with a ratio specified by r.
    """
    data_dir = BASE_PATH/"data"/data_path
    os.chdir(str(data_dir))

    train_files = [f for f in data_dir.glob("train*.txt")]
    test_files = [f for f in data_dir.glob("test*.txt")]

    num_samples_file = sum(1 for line in open(train_files[0]))
    size_validation = int(r * num_samples_file)
    start_pt = random.randint(0, num_samples_file - size_validation)

    def to_json(l, file_name):
        """
        Helper function for writing the contents of a list to a json file.
        """
        label = 0
        if "pos" in file_name:
            label = 1
        with open(file_name + ".json", 'w', encoding='utf-8') as file:
            for i, line in enumerate(l):
                file.write(json.dumps({"id": i, "x": line, "y": label}))
                file.write("\n")
        del l

    for f in train_files:
        print(str(f))
        with open(str(f), 'r', encoding='utf-8') as file:
            train_lines = []
            val_lines = []
            train_lines = list(mitools.islice(file, 0, start_pt))
            val_lines = list(mitools.islice(
                file, start_pt, start_pt + size_validation))
            train_lines.extend(
                list(mitools.islice(file, start_pt + size_validation, num_samples_file)))

            to_json(train_lines, os.path.splitext(str(f))[0])
            if "neg" in str(f):
                to_json(val_lines, str(data_dir/"val_neg"))
            else:
                to_json(val_lines, str(data_dir/"val_pos"))

    for f in test_files:
        with open(str(f), 'r', encoding='utf-8') as file:
            test_lines = file.readlines()
            to_json(test_lines, os.path.splitext(str(f))[0])
    os.chdir("../../")


def create_val_raw_reviews(data_path, r=0.1):
    """
    Creates a validation set of movie reviews from the training set.
    Takes as input the name of the folder where the data is located, as well as
    a value r < 1 that indicates what fraction of the training data to use for
    cross-validation.
    """
    data_dir = BASE_PATH/"data"/data_path
    os.chdir(str(data_dir))
    pos_files = [f for f in data_dir.glob("train_pos.txt")]
    neg_files = [f for f in data_dir.glob("train_neg.txt")]
    num_samples_file = sum(1 for line in open(pos_files[0]))
    length_val = int(r * num_samples_file)
    g = random.randint(0, num_samples_file - length_val)
    for f in pos_files:
        with open(str(f), "rw+", encoding='utf-8') as file:
            val_lines = list(mitools.islice(file, g, g+length_val))
            with open('validation_pos.txt', 'w', encoding='utf-8') as val_file:
                for line in val_lines:
                    val_file.write(line)
    for f in neg_files:
        with open(str(f), "rw+", encoding='utf-8') as file:
            val_lines = list(mitools.islice(file, g, g+length_val))
            with open('validation_neg.txt', 'w', encoding='utf-8') as val_file:
                for line in val_lines:
                    val_file.write(line)
    os.chdir("../../")


def make_wordvec_matrix(text, wordvec_file=WORD_VEC_FILE, max_seq_length=MAX_SEQ_LENGTH):
    wordvec_df = load_vector_embeddings(WORD_VEC_FILE)
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
            # returns vector for unknown words
            return list(np.random.rand(WORDVEC_LENGTH))

    # loops through the columns of the text matrix and replaces each word with it's respective word vector embedding
    for i in range(wordvec_matrix.shape[1]):
        wordvec_matrix[i] = wordvec_matrix.apply(to_wordvec, axis=1)

    print("wordvec_matrix created for input data")

    return wordvec_matrix


def get_split_data(data, labels, train_split, test_split, cv_split):
    """
    Helper function for splitting data into training, cross_validation, and test.
    Data splits must be given in decimal form, e.g. train_split = 0.8, test_split = 0.1, cv_split = 0.1

    split_data(data, 0.7, 0.2, 0.1) would return 70% train, 20% test, 10% cross validation
    """
    assert train_split + test_split + \
        cv_split <= 1.0, "Total split cannot include more than 100% of trainig data."
    num_samples = np.shape(data)[0]
    shuffle_in_unison(data, labels)
    splitpoint_a = math.floor(train_split*num_samples)
    splitpoint_b = math.floor((train_split + test_split)*num_samples)

    x_train, y_train = data[0:splitpoint_a], labels[0:splitpoint_a]
    x_test, y_test = data[splitpoint_a:splitpoint_b], labels[splitpoint_a:splitpoint_b]
    x_cv, y_cv = data[splitpoint_b:], labels[splitpoint_b:]

    return (x_train, y_train, x_test, y_test, x_cv, y_cv)
