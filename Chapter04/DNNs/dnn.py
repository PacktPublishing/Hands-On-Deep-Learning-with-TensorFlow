# TF made EZ
import tensorflow.contrib.learn as learn
from tensorflow.contrib.learn.python.learn.estimators import estimator

# Some basics
import numpy as np
import math
import matplotlib.pyplot as plt
plt.ion()
# Learn more sklearn
# scikit-learn.org
import sklearn
from sklearn import metrics

# Seed the data
np.random.seed(42)
# Load data
data = np.load('data_with_labels.npz')
train = data['arr_0']/255.
labels = data['arr_1']

# Split data into training and validation
indices = np.random.permutation(train.shape[0])
valid_cnt = int(train.shape[0] * 0.1)
test_idx, training_idx = indices[:valid_cnt],\
 indices[valid_cnt:]
test, train = train[test_idx,:],\
              train[training_idx,:]
test_labels, train_labels = labels[test_idx],\
                            labels[training_idx]
 
train = np.array(train,dtype=np.float32)
test = np.array(test,dtype=np.float32)
train_labels = np.array(train_labels,dtype=np.int32)
test_labels = np.array(test_labels,dtype=np.int32)

# Convert features to learn style
feature_columns = learn.infer_real_valued_columns_from_input(train.reshape([-1,36*36]))

# Logistic Regression
classifier = estimator.SKCompat(learn.LinearClassifier(
feature_columns = feature_columns,
n_classes=5))

# One line training
# steps is number of total batches
# steps*batch_size/len(train) = num_epochs
classifier.fit(train.reshape([-1,36*36]),
 train_labels,
 steps=1024,
 batch_size=32)
 
# sklearn compatible accuracy
test_probs = classifier.predict(test.reshape([-1,36*36]))
sklearn.metrics.accuracy_score(test_labels,
 test_probs['classes'])

# Dense neural net
classifier = estimator.SKCompat(learn.DNNClassifier(
 feature_columns = feature_columns,
 hidden_units=[10,5],
 n_classes=5,
 optimizer='Adam'))

# Same training call
classifier.fit(train.reshape([-1,36*36]),
 train_labels,
 steps=1024,
 batch_size=32)
# simple accuracy
test_probs = classifier.predict(test.reshape([-1,36*36]))
sklearn.metrics.accuracy_score(test_labels,
 test_probs['classes'])
# confusion is easy
train_probs = classifier.predict(train.reshape([-1,36*36]))
conf = metrics.confusion_matrix(train_labels,
 train_probs['classes'])
print(conf)