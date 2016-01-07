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

chars74k_root = 'English/'

class Detection(object):
    def __init__(self):
        self.tester = 100
            
    def data_loader(self):
        '''This function loads data (specifically chars74k)
        This function also does grayscale conversion
        '''
        data = pd.read_csv(script_root + 'LISTFILE.txt', sep = ' ', header = None)
        
        data_x = np.zeros((data.shape[0], 1, 32, 32)).astype('float32')
        data_y = np.zeros((data.shape[0], )).astype('int32')
        
        for idx, path in enumerate(data[0]):
            img = imread(data_root + chars74k_root + path)
            img = imresize(img, (32, 32))
            
            if len(img.shape) == 3:
                #TODO check rgb->grey conversion value
                data_x[idx, ...] = img.dot([0.299, 0.587, 0.144])
            elif len(img.shape) == 2:
                data_x[idx, ...] = img
            else:
                raise
                
            return (data_x, data_y)
        
    def visualize(self):
        '''This function visualizes data'''
        i = random.randrange(0, self.data[0].count())
        img = imread(data_root + chars74k_root + self.data[0][i])
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
        
    def preprocess(self, dataset):
        '''This function normalizes data
        Constraints: It consider data_x is passed (i.e. without labels)
        '''
        #TODO integer division or float division
        proc_data = dataset / dataset.std(axis = None)
        proc_data -= proc_data.mean()
        
        return proc_data
        
    def augment_creator(self, dataset):
        '''This function augments data
        '''
        data1_x = shiftup(dataset)
        data2_x = shiftdown(dataset)
        data3_x = shiftleft(dataset)
        data4_x = shiftright(dataset)
        
        data1_y = np.zeros((data1_x.shape[0], )).astype('int')
        data2_y = np.zeros((data2_x.shape[0], )).astype('int')
        data3_y = np.zeros((data3_x.shape[0], )).astype('int')
        data4_y = np.zeros((data4_x.shape[0], )).astype('int')
        
        data_x = np.vstack((data1_x, data2_x))
        data_x = np.vstack((data_x, data3_x))
        data_x = np.vstack((data_x, data4_x))
        
        data_y = np.concatenate([data1_y, data2_y, data3_y, data4_y])
        
        return data_x, data_y
