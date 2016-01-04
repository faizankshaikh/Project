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

def printer():
    return 'All is Well'
