import os
import wget
import quandl
import xml.etree.ElementTree as et
import pandas as pd
import numpy as np
import preprocess_data


class stock_crawl_api():

	def download_stock_data(financials_type, filename): # NOTE  financials_type must be given with first letter capitalized. based off of stockrow formatting for file names 
	
		_tickersList = preprocess_data.load_tickers(filename = filename)

		base_path = os.path.dirname(os.path.realpath(__file__))

		for ticker in _tickersList:
			download_dir = os.path.join(base_path,"data\\" + ticker)
		# skips ticker for specific financial data if (financials_type).xlsx already exists
			if os.path.isfile(download_dir + "\\%(file_name)s.xlsx" %{'file_name': financials_type.lower()}):
				print("Requested %(file_name)s data for %(company_ticker)s already downloaded." %{'file_name': financials_type, 'company_ticker': ticker})
				continue
		# format data donwnload location
			url = "http://stockrow.com/api/companies/%(company_ticker)s/financials.xlsx?dimension=MRQ&section=%(spreadsheet_type)s" %{'company_ticker': ticker, 'spreadsheet_type': financials_type}
		
			if not os.path.exists(download_dir):
				os.makedirs(download_dir)

			download_file = wget.download(url, out = download_dir)

		print ("Successfully downloaded %(spreadsheet_type)s data for specified stock tickers." %{'spreadsheet_type': financials_type.lower()})
		rename_stock_files(financials_type.lower(), "financials")

	def rename_stock_files(filename, new_prefix, old_prefix, is_price_data = False):
		base_path = os.path.dirname(os.path.realpath(__file__))
		data_dir = os.path.join(base_path, "data\\")
		if is_price_data  == False:
			for fn in os.listdir(data_dir):
				if fn == filename or fn == "README.txt":
					continue
				stock_files = os.listdir(os.path.join(data_dir, fn))
		# print (stock_files)
				for file in stock_files:
					if file.startswith(old_prefix):
						full_file = os.path.join(data_dir, fn + "\\" + file)
						os.rename(full_file, full_file.replace(old_prefix, new_prefix))
		else:
			for fn in os.listdir(data_dir):
				if fn == filename or fn == "README.txt":
					continue
				stock_files = os.listdir(os.path.join(data_dir, fn))
		# print (stock_files)
				for file in stock_files:
					if file.startswith(old_prefix):
						full_file = os.path.join(data_dir, fn + "\\" + file)
						os.replace(full_file, full_file.replace(old_prefix + "-" + fn, new_prefix))

		print ("Successfully renamed %(old_file)s files for specified stock tickers to %(new_file)s." %{'old_file': old_prefix.lower(), 'new_file': new_prefix.lower()})

	def download_stock_prices(filename):

		_tickersList = preprocess_data.load_tickers(filename = filename)

		base_path = os.path.dirname(os.path.realpath(__file__))

		# opens api_key.txt which contains unique api key for Quandl dataset calls
		API_FILE = open(base_path + "\\" + "api_key.txt", "r")
		api_key = API_FILE.read()

		for ticker in _tickersList:
			download_dir = os.path.join(base_path, "data\\" + ticker)

			# skips ticker if stock price data already exists
			if os.path.isfile(download_dir + "\\transformed_prices.csv"):
				print("Requested historical price data for %(company_ticker)s already downloaded." %{'company_ticker': ticker})
				continue

			# downloads quandl data using qunadl api
			_price_data = quandl.get("WIKI/" + ticker, end_date = "2018-3-27", transformation = "rdiff", returns = "numpy")
			data_frame = pd.DataFrame(_price_data)
			data_frame.to_csv(download_dir + "\\" + "transformed_prices.csv", index = False)
			# np.savetxt(download_dir + "\\" + "transformed_prices.csv", _price_data, delimiter=",")

		print ("Succesfully downloaded historical price data for specified stock tickers.")

	def stock_data_to_csv(financials_type, filename):
		_tickersList = preprocess_data.load_tickers(filename)

		base_path = os.path.dirname(os.path.realpath(__file__))

		for ticker in _tickersList:
			data_dir = os.path.join(base_path, "data\\" + ticker)

			if os.path.isfile(data_dir + "\\" + financials_type.lower() + ".csv"):
				print("%(financials_type)s data for %(company_ticker)s is already in .csv format." %{"financials_type": financials_type, "company_ticker": ticker})
				continue

			data_frame = pd.read_excel(data_dir + "\\" + financials_type.lower() + ".xlsx", ticker, index_col = None)
			data_frame.to_csv(data_dir + "\\" + financials_type.lower() + ".csv", encoding = "utf-8")

		print ("Conversions from .xlsx to .csv succesful for " + financials_type.lower() + " data.")
