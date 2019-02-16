import time
import warnings
import numpy as np
import os
from os.path import join, dirname
from pathlib import Path
from numpy import newaxis
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import backend as K
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import preprocess_data
import datetime
from preprocess_data import *
# base_path = dirname(os.path.realpath(__file__))
# SNAPSHOT_PREFIX = join(base_path, "models//")
BASE_PATH = Path.cwd()
SNAPSHOT_PREFIX = BASE_PATH/"models"

# DEFAULT TESTING VALUES
batch_size = 500
lstm_units = 25
num_classes = 2
max_epochs = 50

class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        # self.confusion = []
        # self.precision = []

        # self.recall = []
        # self.f1s = []
        # self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.auc.append(sklm.roc_auc_score(targ, score))
        # self.confusion.append(sklm.confusion_matrix(targ, predict))
        # self.precision.append(sklm.precision_score(targ, predict, average=None))
        # self.recall.append(sklm.recall_score(targ, predict))
        # self.f1s.append(sklm.f1_score(targ, predict))
        # self.kappa.append(sklm.cohen_kappa_score(targ, predict))

        return

def train_multi_models(vn, lstm_1_size, drop_1_size, epoch_size):
    # vn = 1
    # lstm_1_size = [10]
    # drop_1_size = [0.75]
    # epoch_size = [200]
    date = datetime.datetime.now()
    raw_data, labels = load_imdb("imdb_reviews")

    for l in lstm_1_size:
        for d in drop_1_size:
            for e in epoch_size:
                try:
                    train_lstm(make_wordvec_matrix(raw_data), labels, "{0}-{1}-{2}_lstm_model".format(date.month, date.day, date.year) + str(vn) + "_f1_" + str(e), batch_size, e, l, d)
                    vn +=1
                except:
                    continue

def train_lstm(data, labels, snapshot_filename, batch_size, max_epochs, lstm_1, drop_1):
    input_dim = np.shape(data)
    label_dim = np.shape(labels)
    assert len(input_dim) == 3, "Input data must have dimensions [num_samples, num words in sample, dimension of word vectors]"
    assert len(label_dim) == 2, "Data labels must have dimensions [num_samples, num_labels]"
    # helpful variables to define in order to determine input_shape
    sample_text_len = input_dim[1]
    word_vector_dimension = input_dim[2]
    num_labels = label_dim[1]
    # TEMPORARY DEFINITIONS FOR TRAINING, CV, AND TEST SPLITS -> change to pass in as func args
    train_split = 0.7
    cv_split = 0.2
    test_split = 0.1
    # split data into train, cv, and test using helper function from preprocess_data
    x_train, y_train, x_test, y_test, x_cv, y_cv = get_split_data(data, labels, train_split, test_split, cv_split)
    # determine input shape for lstm
    input_shape = (sample_text_len, word_vector_dimension)

    # create a tensorboard callback file and log directory
    tensorboard_path_dir = SNAPSHOT_PREFIX/"logs"/snapshot_filename
    if not tensorboard_path_dir.exists():
        tensorboard_path_dir.mkdir(exist_ok=True, parents=True)

    tbCallBack = keras.callbacks.TensorBoard(
        log_dir= tensorboard_path_dir,
        histogram_freq=10,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None)

    # setup LSTM architecture
    # K.set_learning_phase(1)#set learning phase

    model = Sequential()
    model.add(LSTM(units = lstm_1, input_shape = input_shape))
    model.add(Dropout(drop_1))
    print(model.output_shape)
    model.add(Dense(num_classes))
    print(model.output_shape)
    model.add(Activation('softmax'))
    print(model.output_shape)

    metrics_test = Metrics()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    model.fit(
        x_train,
        y_train,
        batch_size = batch_size,
        epochs = max_epochs,
        verbose=1,
        validation_data=(x_cv, y_cv),
        callbacks=[tbCallBack])

    score = model.evaluate(x_test, y_test, verbose=0)
    score_train = model.evaluate(x_cv, y_cv, verbose=0)
    score_cv = model.evaluate(x_train, y_train, verbose=0)

    print('Train loss:', score_train[0])
    print('Train accuracy:', score_train[1])
    print('Val loss:', score_cv[0])
    print('Val accuracy:', score_cv[1])
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # saves the model along with accompanying text detailing train, test, and cv scores
    if not SNAPSHOT_PREFIX.exists():
        SNAPSHOT_PREFIX.mkdir(exist_ok=True, parents=False)
    model.save(str(SNAPSHOT_PREFIX/snapshot_filename/'.h5'))

    snapshot_text = open(SNAPSHOT_PREFIX + snapshot_filename + '.txt', 'w')
    snapshot_text.write(snapshot_filename + ' information:\n')
    snapshot_text.write('Max length of text = ' + str(sample_text_len) + '\n')
    snapshot_text.write('Size of word vectors = ' + str(word_vector_dimension) + '\n')
    snapshot_text.write('Batch size = ' + str(batch_size) + '\n')
    snapshot_text.write('Epoch size = ' + str(max_epochs) + '\n')
    snapshot_text.write('Model architecture: \n')
    snapshot_text.write('lstm_1 = ' + str(lstm_1) + '\n')
    snapshot_text.write('drop_1 = ' + str(drop_1) + '\n')
    snapshot_text.write('\n')
    snapshot_text.write('Model Training Results: \n')
    # snapshot_text.write('Accuracy metric = ' + str(accuracy_metric))
    snapshot_text.write('Train Loss = ' + str(score_train[0]) + '\n')
    snapshot_text.write('Train Accuracy = ' + str(score_train[1]) + '\n')
    snapshot_text.write('Val Loss = ' + str(score_cv[0]) + '\n')
    snapshot_text.write('Val Accuracy = ' + str(score_cv[1]) + '\n')
    snapshot_text.write('Test Loss = ' + str(score[0]) + '\n')
    snapshot_text.write('Test Accuracy = ' + str(score[1]) + '\n')
