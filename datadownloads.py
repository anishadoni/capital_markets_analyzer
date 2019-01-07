import json
import zipfile
import os
import kaggle
import subprocess
from pathlib import Path

BASE_PATH = Path.cwd()
imdb_dir = BASE_PATH/"data"/"imdb_reviews"
glove_dir = BASE_PATH/"data"/"glove_wordvec"

def get_data():
	# helper function for extracting zip files from kaggle downloads
	def zip_files():
		for file in os.listdir():
			zip_ref = zipfile.ZipFile(file, 'r')
			zip_ref.extractall()
			zip_ref.close()

	if not imdb_dir.exists():
		imdb_dir.mkdir(exist_ok=True, parents=True)
	os.chdir("data/imdb_reviews")
	subprocess.run(["kaggle", "datasets", "download", "-d", "anishadoni/dataset-of-50000-imdb-reviews"])
	zip_files()
	os.chdir("../")
	if not glove_dir.exists():
		glove_dir.mkdir(exist_ok=True, parents=True)
	os.chdir("data/glove_wordvec")
	subprocess.run(["kaggle", "datasets", "download", "-d", "anindya2906/glove6b"])
	zip_files()
	os.chdir('../../')
