import os
import numpy as np
import csv
import pandas as pd
import stock_crawler as crawler


######## WARNING - SentimentAnalyzer Class currently under development CURRENTLY UNDER DEVELOPMENT ###########
class SentimentAnalyzer:
	def __init__(self):
		pass

	def model

	def build_model():
		model = Sequential()

		model.add()

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



# instatiates main model
def build_model():
	model = Sequential()

	model.add(CuDNNLSTM(units = 30))
	model.add(Dropout(.2))
	model.add(Dense(units = 30, activation = "relu"))
	model.add(Dense(units = 20, activation = "relu"))

if __name__ == "__main__":

