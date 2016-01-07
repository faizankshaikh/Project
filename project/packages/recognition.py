#TODO Write module 2 here
# Import all modules
import random
import pylab
import pickle as pkl
import numpy as np
import pandas as pd
from scipy.misc import imread, imresize
from lasagne import layers
from theano.tensor.nnet import softmax
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.misc import imread as ims

script_root = '/home/shraddha/Project/scripts/'
data_root = '/home/shraddha/Project/datasets/'
model_root = '/home/shraddha/Project/models/'

class Recognition(object):

    def load_dataset(self):
        self.data = pd.read_csv(script_root + 'LISTFILE.txt', sep = ' ', header = None)
    
    def visualize_dataset(self):
        self.Load_dataset()
        i = random.randrange(0, self.data[0].count())
        img = ims(data_root + 'English/' + self.data[0][i])
        pylab.imshow(img)
        pylab.show()
 
