from __future__ import division

import os
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
        pass
            
    def data_loader(self):
        '''This function loads data (specifically chars74k)
        This function also does grayscale conversion
        '''
        data = pd.read_csv(script_root + 'LISTFILE.txt', sep = ' ', header = None)
        
        data_x = np.zeros((data.shape[0], 1, 32, 32)).astype('float32')
        data_y = np.ones((data.shape[0], )).astype('int32')
        
        for idx, path in enumerate(data[0]):
            img = imread(data_root + chars74k_root + path)
            img = imresize(img, (32, 32))
            
            if len(img.shape) == 3:
                #TODO check rgb->grey conversion value
                data_x[idx, ...] = img.dot([0.299, 0.587, 0.144])
            else:
                data_x[idx, ...] = img
                
        return (data_x, data_y)
        
    def visualize(self, dataset):
        '''This function visualizes data
        
        Input : numpy array (image_number, color_channels, height, width)
        '''
        i = random.randrange(0, dataset.shape[0])
        img = np.reshape(dataset[i, ...], ( dataset.shape[2], dataset.shape[3] ))
        pylab.imshow(img)
        pylab.gray()
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
        proc_data = dataset / dataset.std(axis = None)
        proc_data -= proc_data.mean()
        
        return proc_data
        
    def augment_creator(self, dataset):
        '''This function augments data
        '''
        data1_x = self.shiftup(dataset)
        data2_x = self.shiftdown(dataset)
        data3_x = self.shiftleft(dataset)
        data4_x = self.shiftright(dataset)
        
        data1_y = np.zeros((data1_x.shape[0], )).astype('int')
        data2_y = np.zeros((data2_x.shape[0], )).astype('int')
        data3_y = np.zeros((data3_x.shape[0], )).astype('int')
        data4_y = np.zeros((data4_x.shape[0], )).astype('int')
        
        data_x = np.vstack((data1_x, data2_x))
        data_x = np.vstack((data_x, data3_x))
        data_x = np.vstack((data_x, data4_x))
        
        data_y = np.concatenate([data1_y, data2_y, data3_y, data4_y])
        
        return data_x, data_y
    
    def net_setter(self):
        '''This function sets the architecture of CNN
        '''
        cnn_layers = [
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv3', layers.Conv2DLayer),
            ('hidden4', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ]
        
        self.net = NeuralNet(
            layers = cnn_layers,
            input_shape = (None, 1, 32, 32),
            conv1_num_filters = 32, conv1_filter_size = (5, 5),
            pool1_pool_size = (2, 2),
            dropout1_p = 0.2,
            conv2_num_filters = 64, conv2_filter_size = (5, 5),
            pool2_pool_size = (2, 2),
            dropout2_p = 0.2,
            conv3_num_filters = 128, conv3_filter_size = (5, 5),
            hidden4_num_units = 128,
            output_num_units = 2, output_nonlinearity = softmax,

            batch_iterator_train = BatchIterator(batch_size = 1000),
            batch_iterator_test = BatchIterator(batch_size = 1000),

            update=nesterov_momentum,
            update_learning_rate = 0.01,
            update_momentum = 0.9,

            use_label_encoder = True,
            regression = False,
            max_epochs = 50,
            verbose = 1,
        )

    def train_setter(self, positive_x, negative_x, positive_y, negative_y):
        '''This function combines positive and negative data
        '''
        X = np.vstack((positive_x, negative_x)).astype('float32')
        y = np.concatenate(([positive_y, negative_y])).astype('int32')
        
        return X, y
            
    def trainer(self, X, y):
        '''This function trains CNN on data
        '''
        self.net.fit(X, y)
        
    def predicter(self, X, option = 1):
        '''This function gives prediction
        '''
        if option:
            pred = self.net.predict(X)
        else:
            pred = self.net.predict_proba(X)
            
    def train_saver(self, save_file):
        #TODO try catch
        f = open(os.path.join(model_root, save_file), 'wb')
        pkl.dump(self.net, f)
        f.close()
        
    def generate_heatmap(self, image, prediction, patch_0, patch_1, option = 1):
        '''This function plots a text saliency map
        '''
        if option:
            heatmap = prediction[:, 1].reshape((patch_0, patch_1))
        else:
            heatmap = prediction[:, 0].reshape((patch_0, patch_1))
            
        pylab.pcolor(heatmap[::-1])
        pylab.axis('off')
        pylab.show()
        pylab.imshow(image)
        pylab.show()
