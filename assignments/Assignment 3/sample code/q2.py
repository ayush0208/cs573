import pandas as pd 
import numpy as np 
from collections import defaultdict
import re
import sys
import matplotlib.pyplot as plt
import random
import math
from scipy import optimize as op  
from numpy.linalg import norm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# importing data
df1 = pd.read_csv( sys.argv[1], encoding='ISO-8859-1') #training set
df2 = pd.read_csv( sys.argv[2], encoding='ISO-8859-1') #test set
#print df2.shape
modelIdx = sys.argv[3]
#print "modelIdx",modelIdx
##########..................LR Scratch .....................
def sigmoid(z):
	    return(1 / (1 + np.exp(-z)))  # z = w.T.dot(x_i)

def cost_func(features, target, weights):
    scores = np.dot(features, weights)
    cf = np.sum( target*scores - np.log(1 + np.exp(scores)) )  +  _lambda/(2) * sum(weights**2)
    return cf

def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
    	#print "features before",features.shape
        intercept = np.ones((features.shape[0], 1))
        #print "intercept", intercept.shape
        features = np.hstack((intercept, features))
        #print "features after",features.shape
    
    weights = np.zeros(features.shape[1])
    #print "weights loop", weights.shape
   
    
    for step in xrange(num_steps):
    	w1 = np.array(weights)
    	#print "w1", w1
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
       	output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal) +  (_lambda * weights)
        weights += learning_rate * gradient
        w2 = np.array(weights)
        #print "w2", w2, type(w2)
        diff = np.subtract(w2, w1)
        #print "diff", diff
        l2 = norm(diff)
        #print "l2", l2
        if l2 < 0.000001:
        	break
        
    	
        # Print log-likelihood every so often
        #if step % 10 == 0:
            #print cost_func(features, target, weights)
    return weights
if modelIdx == '1': #LR

	X_train = df1.values[0:, 1:261]
	Y_train = df1.values[0:, 261:262]
	#print "X_train", type(X_train), X_train
	#print Y_train, len(Y_train)
	#y = (df1.iloc[:, 261:262]).values
	#print y, len(y)
	#y_train = y_train.ravel()
	X_test = df2.values[0:, 1:261]
	Y_test = df2.values[0:, 261:262]
	#y_test = y_test.ravel()



	#Logistic Regression

	train_features = X_train
	train_labels = Y_train.reshape(Y_train.shape[0],)
	test_label = Y_test.reshape(Y_test.shape[0],)
	_lambda = 0.01 

	#learning part
	#step_size = learning rate
	weights = logistic_regression(train_features, train_labels, num_steps = 500, learning_rate = 0.01, add_intercept=True)
	#print "weights", weights.shape
	# prediction for training
	data_with_intercept = np.hstack((np.ones((train_features.shape[0], 1)),train_features))
	#print "data_with_intercept ",data_with_intercept.shape
	final_scores = np.dot(data_with_intercept, weights)
	preds = np.round(sigmoid(final_scores))
	train_accuracy_LR = (preds == train_labels).sum().astype(float) / len(preds)
	print 'Training Accuracy LR:', round(train_accuracy_LR,2)

	# prediction for testing 
	data_with_intercept2 = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
	final_scores2 = np.dot(data_with_intercept2, weights)
	preds2 = np.round(sigmoid(final_scores2))
	test_accuracy_LR = (preds2 == test_label).sum().astype(float) / len(preds2) 
	print 'Testing Accuracy LR:',round(test_accuracy_LR,2)



######### SVM from scratch ###########
def svm(features, target, num_steps2, learning_rate, add_intercept = False):
    if add_intercept:
    	#print "features before",features.shape
        intercept = np.ones((features.shape[0], 1))
        #print "intercept", intercept.shape
        features = np.hstack((intercept, features))
        #print "features after",features.shape
        
    weights = np.zeros(features.shape[1])
    out = []
    N = target.shape[0]
    #print N
    for step in range (0,num_steps2):
   		#print "step", step
   		for i, val in enumerate(features):
   			w1 = np.array(weights)
   			val1 = np.dot(features[i],weights)
   			#print val1
   			if (target[i] * val1) < 1.0:
   				#weights = weights + (learning_rate * ((target[i] * features[i]) - (2 *_lambda * weights)))
   				weights = weights + (learning_rate * (((target[i] * features[i]) - (1 *_lambda * weights))/N))
   			else:
   				#weights = weights + (learning_rate * (- (2 *_lambda * weights)))
   				weights = weights + (learning_rate * (- ((1 *_lambda * weights)/N)))
   			w2 = np.array(weights)
	        #print "w2", w2, type(w2)
	        diff = np.subtract(w2, w1)
	        #print "diff", diff
	        l2 = norm(diff)
	        #print "l2", l2
	        if l2 < 0.000001:
	        	break
    
    return weights

if modelIdx == '2': #Linear SVM
	X_train = df1.values[0:, 1:261]
	Y_train = df1.values[0:, 261:262]
	#print "Y_train",Y_train
	#c = 0
	for i,val in enumerate(Y_train):
		if Y_train[i] == 0:
			Y_train[i] = -1
			#c+=1
	#print c
	#print Y_train
	#print "X_train", type(X_train), X_train
	X_test = df2.values[0:, 1:261]
	Y_test = df2.values[0:, 261:262]
	#print "Y_test", Y_test
	for i,val in enumerate(Y_test):
		if Y_test[i] == 0:
			Y_test[i] = -1
	#y_test = y_test.ravel()

	train_features = X_train
	train_labels = Y_train.reshape(Y_train.shape[0],)
	test_label = Y_test.reshape(Y_test.shape[0],)

	_lambda = 0.01 

	#learning part
	#step_size = learning rate
	#print "X_train.shape", X_train.shape
	weights = svm(train_features, train_labels, num_steps2 = 500, learning_rate = 0.5, add_intercept=False)
	final_scores = np.sign(np.dot(train_features,weights))
	#print final_scores
	'''
	c = 0
	for i,val in enumerate(final_scores):
		if final_scores[i] == -1:
			#final_scores[i] = -1
			c+=1
	print c
	'''
	train_accuracy_SVM = (final_scores == train_labels).sum().astype(float) / len(final_scores)
	print 'Training Accuracy for svm',round(train_accuracy_SVM,2)

	# prediction for testing 
	final_scores2 = np.sign(np.dot(X_test, weights))
	test_accuracy_SVM = (final_scores2 == test_label).sum().astype(float) / len(final_scores2) 
	print 'Testing Accuracy for svm: ', round(test_accuracy_SVM,2)