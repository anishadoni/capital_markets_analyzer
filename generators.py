from keras.utils import Sequence, HDF5Matrix
from pathlib import Path
class DataGenerator(Sequence):
    def __init__(self, x_set_path, y_set_path, batch_size):
    	# x_set_path and y_set_path are the paths to where the datasets for each are stored as a .h5 file
    	self.x, self.y = x_set_path, y_set_path
    	self.batch_size = batch_size

    def __len__():
    	return int(np.ceil(len(HDF5Matrix(self.x/"reviews.h5", "review_x"))/float(self.batch_size)))

    def __getitem__(self, idx):
    	batch_x = HDF5Matrix(self.x/"reviews.h5", "review_x", start=idx*self.batch_size, end=(idx+1)*self.batch_size).value
    	batch_y = HDF5Matrix(self.y/"reviews.h5", "review_y", start=idx*self.batch_size, end=(idx+1)*self.batch_size).value

    	return batch_x, keras.utils.to_categorical(batch_y, num_classes=2)