from CNN.network import *
from CNN.utils import *
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle


parser = argparse.ArgumentParser(description='Train a convolutional neural network.')
parser.add_argument('save_path', metavar = 'Save Path', help='name of file to save parameters in.')

if __name__ == '__main__':
	m =1
	X = extract_data('t10k-images-idx3-ubyte.gz', m, 28)
	y_dash = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
	# Normalize the data
	X-= int(np.mean(X)) # subtract mean
	X/= int(np.std(X)) # divide by standard deviation
	
	test_data = np.hstack((X,y_dash))
	X = test_data[:,0:-1]
	X = X.reshape(len(test_data), 1, 28, 28)
	plt.imshow(X)
	plt.show()
	y = test_data[:,-1]