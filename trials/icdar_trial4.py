import os
import random
import pylab
import time
import csv
import enchant
import pandas as pd
import numpy as np
import cPickle as pkl
from lasagne import layers, updates
from scipy.misc import imread, imresize
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet, BatchIterator
from sklearn.feature_extraction.image import extract_patches

data_root = '/home/faizy/workspace/.project/project/datasets/'
model_root = '/home/faizy/workspace/.project/project/models/'

icdar_root = 'icdar15/'
test_root = 'Challenge2_Test_Task3_Images'

test_size = 1095
alphabet = 'abcdefghijklmnopqrstuvwxyz'
filename = 'sub01.txt'

# load models
f = open(model_root + 'detector_2.pkl', 'rb')
detector = pkl.load(f)
f.close()

f = open(model_root + 'recognizer.pkl', 'rb')
recognizer = pkl.load(f)
f.close()

d = enchant.Dict()

'''
# visualize dataset
i = random.randrange(1, test_size)
img = imread(os.path.join(data_root, icdar_root, test_root, ('word_' + str(i) + '.png')), flatten = True)
pylab.imshow(img)
pylab.gray()
pylab.show()
'''
def main():
    pred = []
    id_arr = []
    start_time = time.time()
    
    for i in range(1, test_size + 1):
        img = imread(os.path.join(data_root, icdar_root, test_root, ('word_' + str(i) + '.png')), flatten = True)
        image_height = img.shape[0]
        image_width = img.shape[1]

        id_arr.append(str(i) + '.png')

        # check for smaller width image
        if image_width > image_height:
            patches = extract_patches(img, (image_height, image_height*0.60))
        else:
            patches = extract_patches(img, (image_height, image_width))

        new_lst = []
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                new_lst.append(imresize(patches[i, j, :, :], (32, 32)))
                

        new_list = np.stack(new_lst)
        tester = new_list.reshape(patches.shape[0]*patches.shape[1], 1, 32, 32).astype('float32')
        tester.shape

        tester /= tester.std(axis = None)
        tester -= tester.mean()
        tester = tester.astype('float32')

        preder = detector.predict_proba(tester)


        heatmap = preder[:, 1].reshape((patches.shape[0], patches.shape[1]))

        predict_signal = np.reshape(heatmap, patches.shape[1]*patches.shape[0])
        
        x_1 = np.arange(patches.shape[1])
        y_1 = np.zeros(patches.shape[1])
        x_2 = np.arange(32, patches.shape[1] + 32)
        y_2 = np.ones(patches.shape[1])
        scores_ = predict_signal
        
        boxes = np.stack((x_1, y_1, x_2, y_2, scores_)).T
        
        def nms(dets, thresh):
            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(ovr <= thresh)[0]
                order = order[inds + 1]

            return np.sort(np.array(keep, dtype = int))
            
        peakind = nms(boxes,thresh=0)

        word = np.zeros((len(peakind), 1, 32, 32))
        for idx, item in enumerate(peakind):
            word[idx, ...] = tester[item, 0, :, :] 
            
        word = word.astype('float32')

        predict = recognizer.predict(word)

        def classer(arrayer):
            classer_array = []
            for i in range(len(arrayer)):
                if (0 <= arrayer[i] < 10):
                    classer_array.append(arrayer[i])
                elif (10 <= arrayer[i] < 36) :
                    classer_array.append(alphabet[arrayer[i] - 10].upper())
                elif (36 <= arrayer[i] < 62):
                    classer_array.append(alphabet[arrayer[i] - 36])
                else : 
                    print 'Is the array correct!?'
            return classer_array
        real_pred = classer(predict)
        real_pred = map(str, real_pred)
        letter_stream = ''.join(real_pred)
        
        print letter_stream
                
        if image_width > image_height:
            if d.suggest(letter_stream):
                pred.append(d.suggest(letter_stream)[0])
            else:
                pred.append(letter_stream)    
        else:
            pred.append(letter_stream)
        
    with open('sub01.txt', 'w') as f:
        for l1, l2 in zip(id_arr, pred):
            f.write(l1 + ', ' + '"' + l2 + '"' + '\n')
    
    print "time taken: ", time.time() - start_time
    
if __name__ == '__main__':
    main()
