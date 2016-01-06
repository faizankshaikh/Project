#TODO Write module 1 here

import pylab
import random
import numpy as np
import pandas as pd
import cPickle as pkl
from lasagne import layers
from theano.tensor.nnet import softmax
from scipy.misc import imread, imresize
from lasagne.updates import nesterov_momentum
from sklearn.cross_validation import train_test_split
from nolearn.lasagne import NeuralNet, BatchIterator

script_root = '/home/faizy/workspace/project/project/scripts/'
data_root = '/home/faizy/workspace/project/project/datasets/'
model_root = '/home/faizy/workspace/project/project/models/'

class Detection(object):
    def __init__(self):
        self.tester = 100
            
    def data_loader(self):
        '''This function loads data'''
        self.data = pd.read_csv(script_root + 'LISTFILE.txt', sep = ' ', header = None)
        
    def visualize(self):
        '''This function visualizes data'''
        i = random.randrange(0, self.data[0].count())
        img = imread(data_root + 'English/' + self.data[0][i])
        pylab.imshow(img)
        pylab.show()
        
    def shiftup(self, dataset):
        '''This function is for data augmentation
        Take an image and shift it up
        '''
        shifted_dataset = np.zeros(dataset.shape)
        # loop for images
        for i in range(dataset.shape[0]):
            # loop for shift up
            for j in range(16):
                shifted_dataset[i, 0, j:j+1, :] = dataset[i, 0, 16 + j : 16 + j + 1, :]
            for j in range(16, 32):
                shifted_dataset[i, 0, j:j+1, :] = shifted_dataset[i, :, 15, :]
        return shifted_dataset
        
    def shiftdown(self, dataset):
        '''This function is for data augmentation
        Take an image and shift it down
        '''
        shifted_dataset = np.zeros(dataset.shape)
        # loop for images
        for i in range(dataset.shape[0]):
            # loop for shift up
            for j in range(16, 32):
                shifted_dataset[i, 0, j:j+1, :] = dataset[i, 0, j - 16 : j + 1 - 16, :]
            for j in range(16):
                shifted_dataset[i, 0, j:j+1, :] = shifted_dataset[i, :, 16, :]
        return shifted_dataset
        
    def shiftleft(self, dataset):
        '''This function is for data augmentation
        Take an image and shift it left
        '''
        shifted_dataset = np.zeros(dataset.shape)
        # loop for images
        for i in range(dataset.shape[0]):
            for j in range(16):
                shifted_dataset[i, 0, :, j:j+1] = dataset[i, 0, :, 16 + j: 16 + j + 1]
            for j in range(16, 32):
                shifted_dataset[i, :, :, j] = shifted_dataset[i, :, :, 15]
        
        return shifted_dataset
        
    def shiftright(self, dataset):
        '''This function is for data augmentation
        Take an image and shift it right
        '''
        shifted_dataset = np.zeros(dataset.shape)
        # loop for images
        for i in range(dataset.shape[0]):
            for j in range(16, 32):
                shifted_dataset[i, 0, :, j : j + 1] = dataset[i, 0, :, j - 16 : j + 1 - 16]
            for j in range(16):
                shifted_dataset[i, 0, :, j] = dataset[i, 0, :, 15]
        
        return shifted_dataset
