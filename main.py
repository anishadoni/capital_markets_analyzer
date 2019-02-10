import numpy as np
import pandas as pd
from pathlib import Path
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from generators import DataGenerator
import stock_crawler as crawler


# Useful constants
BASE_PATH = Path.cwd()

def load_data(filename):
	_tickersList = crawler.load_tickers(filename = filename)
	_growth_data = []
	_metrics_data = []
	_price_data = []

	base_path = os.path.dirname(os.path.realpath(__file__))

	for ticker in _tickersList:
		data_dir = os.path.join(base_path, "data\\" + ticker)


		# test code using pandas to read csv data
		temp_price_data = pd.read_csv(
			data_dir + "\\transformed_prices.csv",
			index_col = 0,
			usecols = ['Date', 'Close']
			)

		# .T transposes the data to match format of prices (date is in rows)
		temp_growth_data = pd.read_csv(
			data_dir + "\\growth.csv",
			index_col = 0).T

		temp_metrics_data = pd.read_csv(
			data_dir + "\\metrics.csv",
			index_col = 0).T

		_price_data.append(temp_price_data)
		_growth_data.append(temp_growth_data)
		_metrics_data.append(temp_metrics_data)

	print ("Price, growth, and metrics data successfully loaded.")
	return (_price_data, temp_growth_data, temp_metrics_data)

def get_model(shape):
	"""
	Returns a keras model
	"""
	input_shape = shape
	model = Sequential()
    model.add(CuDNNLSTM(units = 30, input_shape = input_shape))
    model.add(Dropout(drop_1))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

	return model

def train_model():
	train_generator = DataGenerator(BASE_PATH/"data"/"imdb_reviews", BASE_PATH/"data"/"imdb_reviews", 100)
	

if __name__ == "__main__":

