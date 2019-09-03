import numpy as np
import pandas as pd
from pathlib import Path
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, CuDNNLSTM
from keras.models import Sequential
from keras import metrics
from generators import DataGenerator
from imdb import *
import stock_crawler as crawler
import json
import os
import datetime


# Useful constants
BASE_PATH = Path.cwd()
IMDB_DATA_PATH = BASE_PATH/"data"/"imdb_reviews"
INPUT_SHAPE = (2470, WORDVEC_LENGTH)
MODELS_PATH = BASE_PATH/"models"


def load_data(filename):
    _tickersList = crawler.load_tickers(filename=filename)
    _growth_data = []
    _metrics_data = []
    _price_data = []

    base_path = os.path.dirname(os.path.realpath(__file__))

    for ticker in _tickersList:
        data_dir = os.path.join(base_path, "data\\" + ticker)

        # test code using pandas to read csv data
        temp_price_data = pd.read_csv(
            data_dir + "\\transformed_prices.csv",
            index_col=0,
            usecols=['Date', 'Close']
        )

        # .T transposes the data to match format of prices (date is in rows)
        temp_growth_data = pd.read_csv(
            data_dir + "\\growth.csv",
            index_col=0).T

        temp_metrics_data = pd.read_csv(
            data_dir + "\\metrics.csv",
            index_col=0).T

        _price_data.append(temp_price_data)
        _growth_data.append(temp_growth_data)
        _metrics_data.append(temp_metrics_data)

    print("Price, growth, and metrics data successfully loaded.")
    return (_price_data, temp_growth_data, temp_metrics_data)


def get_model(shape, drop_1=0.2, num_classes=2):
    """
    Returns a keras model
    """
    input_shape = shape
    model = Sequential()
    model.add(CuDNNLSTM(units=30, input_shape=input_shape))
    model.add(Dropout(drop_1))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def load_model_dict():
    model_dict = {}
    with open("models.json", 'r', encoding='utf-8') as file:
        if os.stat("models.json").st_size != 0:
            for line in file:
                model_dict = json.loads(line)
        return model_dict


def save_model(model, filename, vn):
    if not MODELS_PATH.exists():
        MODELS_PATH.mkdir()

    model_dict = load_model_dict()
    with open("models.json", 'w', encoding='utf-8') as file:
        if len(model_dict) == 0:
            model_dict = {vn: filename}
            file.write(json.dumps(model_dict) + "\n")
        else:
            for line in file:
                model_dict[vn] = filename + "_" + str(vn)
                file.write(json.dumps(model_dict) + "\n")
    model.save(str(MODELS_PATH/model_dict[vn]) + '.h5')


def train_model(use_multiprocessing=false):
    date = datetime.datetime.now()
    model_filename = "{0}-{1}-{2}_lstm_model".format(
        date.month, date.day, date.year)

    train_generator = DataGenerator(IMDB_DATA_PATH, IMDB_DATA_PATH, 100)
    val_generator = DataGenerator(IMDB_DATA_PATH, IMDB_DATA_PATH, "val", 100)
    test_generator = DataGenerator(IMDB_DATA_PATH, IMDB_DATA_PATH, "test", 100)
    model = get_model(INPUT_SHAPE)

    # Creates a directory to save tensorboard callback logs.
    tensorboard_path_dir = BASE_PATH/"logs"/model_filename
    if not tensorboard_path_dir.exists():
        tensorboard_path_dir.mkdir(exist_ok=True, parents=True)

    tbCallBack = keras.callbacks.TensorBoard(
        log_dir=str(tensorboard_path_dir),
        histogram_freq=10,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None
    )

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=[f1, metrics.binary_accuracy])
    model.fit_generator(
        generator=train_generator,
        epochs=100,
        callbacks=[tbCallBack],
        validation_data=val_generator,
        validation_freq=10,
        workers=2,
        use_multiprocessing=True,
        shuffle=True,
    )

    score = model.evaluate_generator(
        test_generator, use_multiprocessing=use_multiprocessing, verbose=0)
    score_train = model.evaluate_generator(
        val_generator, use_multiprocessing=use_multiprocessing, verbose=0)
    score_cv = model.evaluate_generator(
        train_generator, use_multiprocessing=use_multiprocessing, verbose=0)
    # needs to be fixed, violates abstraction barriers
    model_dict = load_model_dict()
    vn = 0
    if len(model_dict) == 0:
        vn = 1
    else:
        vn = model_dict.keys()[-1] + 1
    save_model(model, model_filename, vn)


if __name__ == "__main__":
    train_model()
