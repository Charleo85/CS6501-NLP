from nltk.tokenize import word_tokenize
import numpy as np
from random import shuffle
import pickle
from tqdm import tqdm_notebook as tqdm

def save_pickle(data, filename='sample.pickle'):
    with open(filename, mode='wb') as f:
        pickle.dump(data, f)

def normalize(word):
    w = word.lower()
    if w.isdigit(): return "NUM"
    else: return w

tf = {}

def loadData(fname):
	f = open(fname, 'r')
	sentences = f.readlines()
	f.close()
	features = []
	for stn in sentences:
	    feature = []
	    words = word_tokenize(stn)
	    for w in words:
	        v = normalize(w)
	        if v in tf: 
	            tf[v] += 1
	        else:
	            tf[v] = 1
	        feature.append(v)

	    features.append(feature)
	return features

word2index = {}
index2word = {}
index = 1
index2word[0] = 'UNK'
for w, freq in tf.items():
    if w not in word2index:
        if freq < 3:
            word2index[w] = 0
        else:
            word2index[w] = index
            index2word[index] = w
        index += 1

V = index

def loadfxy(labels, features):
    print(len(labels), len(features))
    bow = [None]*len(features)
    for i in range(len(features)):
        feature = np.zeros((V+1, 1), dtype=np.uint8 )
        l = int(labels[i])
        for v in train_features[i]:
            key = 0 #map unknown token to UNK
            if v in word2index:
                key = word2index[v]
            feature[key] += 1
        feature[-1] = 1 #constant
        bow[i] = (feature, l)
    return bow

train_features = loadData('trn.data')
f = open('trn.label', 'r')
train_labels = f.readlines()
f.close()
train_X_Y = loadfxy(train_labels, train_features)
save_pickle(train_X_Y, 'train_X_Y.pickle')


dev_features = loadData('dev.data')
f = open('dev.label', 'r')
dev_labels = f.readlines()
f.close()
dev_X_Y = loadfxy(dev_labels, dev_features)
save_pickle(dev_X_Y, 'dev_X_Y.pickle')

def save_pickle(data, filename='sample.pickle'):
    with open(filename, mode='wb') as f:
        pickle.dump(data, f)

def normalize(word):
    w = word.lower()
    if w.isdigit(): return "NUM"
    else: return w

tf = {}

def loadData(fname):
	f = open(fname, 'r')
	sentences = f.readlines()
	f.close()
	features = []
	for stn in sentences:
	    feature = []
	    words = word_tokenize(stn)
	    for w in words:
	        v = normalize(w)
	        if v in tf: 
	            tf[v] += 1
	        else:
	            tf[v] = 1
	        feature.append(v)

	    features.append(feature)
	return features

train_features = loadData('trn')
print(tf)
print(train_features)
pickle.save(train_features, 'train_tf')
pickle.save(train_features, 'train_features')

word2index = {}
index2word = {}
index = 1
index2word[0] = 'UNK'
for w, freq in tf.items():
    if w not in word2index:
        if freq < 3:
            word2index[w] = 0
        else:
            word2index[w] = index
            index2word[index] = w
        index += 1

V = index

def loadfxy(labels, features):
    print(len(labels), len(features))
    bow = [None]*len(features)
    for i in range(len(features)):
        feature = np.zeros((V+1, 1), dtype=np.uint8 )
        l = int(labels[i])
        for v in train_features[i]:
            key = 0 #map unknown token to UNK
            if v in word2index:
                key = word2index[v]
            feature[key] += 1
        feature[-1] = 1 #constant
        bow[i] = (feature, l)
    return bow

train_features = loadData('trn.data')
f = open('trn.label', 'r')
train_labels = f.readlines()
f.close()
train_X_Y = loadfxy(train_labels, train_features)
save_pickle(train_X_Y, 'train_X_Y.pickle')


dev_features = loadData('dev.data')
f = open('dev.label', 'r')
dev_labels = f.readlines()
f.close()
dev_X_Y = loadfxy(dev_labels, dev_features)
save_pickle(dev_X_Y, 'dev_X_Y.pickle')
