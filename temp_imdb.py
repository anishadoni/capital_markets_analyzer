import pickle
import os
import numpy as np
import glob

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

def combine_files(file_dir):
	data_dir = os.path.join(BASE_PATH, file_dir)
	files = glob.glob(data_dir + "*.txt")
	with open(data_dir + 'combined_files.txt', 'w', encoding = "utf-8") as f:
		for file in files:
			with open(file, 'r', encoding = "utf-8") as new_file:
				review = new_file.read()
				f.write(review + "\n")
				new_file.close()
		f.close()

if __name__ == "__main__":
	combine_files("data\\imdb_reviews\\reviews\\train\\neg\\")