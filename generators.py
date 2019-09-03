from keras.utils import Sequence, HDF5Matrix
from pathlib import Path
import pandas as pd
import numpy as np
import more_itertools as mitools
import keras
import json
import re
import glob

BASE_PATH = Path.cwd()


class DataGenerator(Sequence):
    def __init__(self, x_set_path, y_set_path, flag, batch_size):
        # x_set_path and y_set_path are the paths to where the datasets for each are stored as a .h5 file
        self.x, self.y = x_set_path, y_set_path
        self.batch_size = batch_size
        self.flag = flag

    def __len__(self):
        return int(np.ceil(len(HDF5Matrix(self.x/"reviews.h5", "x_" + self.flag))/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = HDF5Matrix(self.x/"reviews.h5", "x_" + self.flag,
                             start=idx*self.batch_size, end=(idx+1)*self.batch_size).value
        batch_y = HDF5Matrix(self.y/"reviews.h5", "y_" + self.flag,
                             start=idx*self.batch_size, end=(idx+1)*self.batch_size).value

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=2)


# helper function for removing special characters from strings
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def cleanSentences(string):
    """
    Helper function for "cleaning" a string, e.g. removing unrecognized or
    unimportant characters. Returns the "cleaned" version of the string.
    """
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

# NOTE WORK IN PROGRESS


def generate_raw_reviews(data_directory, flag="train", batch_size=100):
    data_dir = BASE_PATH/"data"/data_directory
    pos_files = [f for f in data_dir.glob(flag + "_pos.txt")]
    neg_files = [f for f in data_dir.glob(flag + "_neg.txt")]

    all_files = pos_files + neg_files
    num_samples_file = sum(1 for line in open(all_files[0]))
    for f in all_files:
        with open(str(f), "r", encoding='utf-8') as file:
            while True:
                text_chunk = [list(map(cleanSentences, line.split()))
                              for line in list(mitools.islice(file, batch_size))]
                if not text_chunk:
                    break
                yield text_chunk


def generate_raw_reviews_json(data_directory, flag="train", batch_size=100):
    """
    Generates chunks of data from a json file given the data folder, a flag
    (eg. "train", "test") and a batch_size (default is 100). Assumes that the
    data in the json files is formatted with each line in the file like so:
    {"id": some_id, "x": "some string of txt", "y": some_label}
    Returns a list of these dictionaries with length batch_size, with the string
    value corresponding to "x" converted to a list of tokens with unrecognized
    characters removed.
    """
    data_dir = BASE_PATH/"data"/data_directory
    files = [f for f in data_dir.glob(flag + "*.json")]
    num_samples = sum(1 for line in open(files[0]))

    for f in files:
        with open(str(f), 'r', encoding='utf-8') as file:
            while True:
                text_chunk = []
                for line in list(mitools.islice(file, batch_size)):
                    raw_dict = json.loads(line)
                    raw_dict["x"] = list(
                        map(cleanSentences, raw_dict["x"].split()))
                    text_chunk.append(raw_dict)
                if not text_chunk:
                    break
                yield text_chunk
