import os
import random
import pylab
import time
import csv
import pandas as pd
import numpy as np
import cPickle as pkl
from lasagne import layers, updates
from scipy.misc import imread, imresize
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet, BatchIterator
from sklearn.feature_extraction.image import extract_patches

data_root = '/home/faizy/workspace/project/project/datasets/'
model_root = '/home/faizy/workspace/project/project/models/'

icdar_root = 'icdar15/'
test_root = 'Challenge2_Test_Task3_Images'

test_size = 1095

filename = 'sub01.txt'

# load models
f = open(model_root + 'detector_2.pkl', 'rb')
detector = pkl.load(f)
f.close()

f = open(model_root + 'recognizer.pkl', 'rb')
recognizer = pkl.load(f)
f.close()

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

        id_arr.append('word_' + str(i) + '.png')

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

        a = np.reshape(heatmap, patches.shape[1]*patches.shape[0])

        from scipy.ndimage.filters import maximum_filter
        peakind = np.nonzero(maximum_filter (a, size=(patches.shape[1]/5)*0.75) == a)[0]
        breakind = np.nonzero(maximum_filter((1 - a), size=(patches.shape[1]/5)) == (1 - a))[0]

        word = np.zeros((len(peakind), 1, 32, 32))
        for idx, item in enumerate(peakind):
            word[idx, ...] = tester[item, 0, :, :]
            
        word = word.astype('float32')

        predict = recognizer.predict(word)

        # Define word recognition functions
        import re, collections

        def words(text): return re.findall('[a-z]+', text.lower()) 

        def train(features):
            model = collections.defaultdict(lambda: 1)
            for f in features:
                model[f] += 1
            return model

        NWORDS = train(words(file(data_root + 'big.txt').read()))

        alphabet = 'abcdefghijklmnopqrstuvwxyz'

        def edits1(word):
           splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
           deletes    = [a + b[1:] for a, b in splits if b]
           transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
           replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
           inserts    = [a + c + b     for a, b in splits for c in alphabet]
           return set(deletes + transposes + replaces + inserts)

        def known_edits2(word):
            return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2.lower() in NWORDS)

        def known(words): return set(w for w in words if w.lower() in NWORDS)

        def correct(word):
            candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
            return sorted(candidates,  key=NWORDS.get, reverse = True)

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
        
        def str_corr(letter_stream):
            cnt_lwr=0
            cnt_upr=0
            for i in letter_stream:
                if(i.islower()):
                    cnt_lwr += 1;
                else:
                    cnt_upr +=1;
            if(cnt_lwr > cnt_upr):
                if(letter_stream[0].isupper()):
                     letter_stream = letter_stream.title()
                else:
                     letter_stream = letter_stream.lower()
            else:
                 if(letter_stream[0].isupper()):
                     letter_stream = letter_stream.title()
                 else:
                     letter_stream = letter_stream.upper()
#rint letter_stream
        str_corr(letter_stream)
        

        #print 'Probable word is: ', correct(letter_stream)[0]
        

        pred.append("'" + correct(letter_stream)[0] + "'")
        
    pd.DataFrame({'image': id_arr, 'words' : pred }).to_csv(filename, index = False, header = False, quoting = csv.QUOTE_MINIMAL)
    
    f = open(filename, 'r')
    text = f.read()
    f.close()
    text = text.replace("'", "\"")
    f = open(filename, 'w')
    f.write(text)
    f.close()
    
    print "time taken: ", time.time() - start_time
    
if __name__ == '__main__':
    main()
