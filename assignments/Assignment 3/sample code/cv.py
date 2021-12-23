import pandas as pd 
import numpy as np 
from collections import defaultdict
from numpy.linalg import norm
import re
import sys
import matplotlib.pyplot as plt
import random
import math
from scipy import optimize as op
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#from sklearn import svm
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
# importing training data
df = pd.read_csv( sys.argv[1], encoding='ISO-8859-1') #training set
#print df.shape #ok 5400 * 262
sampling = df.sample(frac=1, random_state = 18)
# partition sample data into 10 disjoint sets
S1 = sampling.values[0:520, 1:262]
S2 = sampling.values[520:1040, 1:262]
S3 = sampling.values[1040:1560, 1:262]
S4 = sampling.values[1560:2080, 1:262]
S5 = sampling.values[2080:2600, 1:262]
S6 = sampling.values[2600:3120, 1:262]
S7 = sampling.values[3120:3640, 1:262]
S8 = sampling.values[3640:4160, 1:262]
S9 = sampling.values[4160:4680, 1:262]
S10 = sampling.values[4680:5200, 1:262]

#print S1.shape, S1
#print  S2.shape, S2
#print S10.shape, S10
'''
def sklearn(train_set, test_set):
    train_feature = train_set.values[0:, 0:260]
    train_label = train_set.values[0:, 260:261]
    test_feature = test_set.values[0:, 0:260]
    test_label2 = test_set.values[0:, 260:261]
    clf = svm.SVC(kernel='linear', C = 1.0, max_iter = 500, tol=0.000001)
    clf.fit(train_feature,train_label)
    print "svm sklearn training accuracy: ", clf.score(train_feature, train_label)
    #Predict Output
    predicted= clf.predict(test_feature)
    print "svm sklearn test accuracy: ", accuracy_score(test_label2, predicted)
    return clf.score(train_feature, train_label), accuracy_score(test_label2, predicted)
'''
def linear_svm(features2, target2, num_steps2, learning_rate2, add_intercept = False):
    if add_intercept:
        #print "features before",features.shape
        intercept2 = np.ones((features2.shape[0], 1))
        #print "intercept", intercept.shape
        features2 = np.hstack((intercept2, features2))
        #print "features after",features.shape
        
    weights2 = np.zeros(features2.shape[1])
    out = []
    N = target2.shape[0]
    #print N
    for step2 in range (0,num_steps2):
        #print "step", step
        for i, val in enumerate(features2):
            w1 = np.array(weights2)
            val12 = np.dot(features2[i],weights2)
            #print val1
            if (target2[i] * val12) < 1.0:
                #weights2 = weights2 + (learning_rate2 * ((target2[i] * features2[i]) - (2 *_lambda * weights2)))
                weights2 = weights2 + (learning_rate2 * (((target2[i] * features2[i]) - (1 *_lambda * weights2))/N))
            else:
                #weights2 = weights2 + (learning_rate2 * (- (2 *_lambda * weights2)))
                weights2 = weights2 + (learning_rate2 * (- ((1 *_lambda * weights2)/N)))
            w2 = np.array(weights2)
            #print "w2", w2, type(w2)
            diff = np.subtract(w2, w1)
            #print "diff", diff
            l2 = norm(diff)
            #print "l2", l2
            if l2 < 0.000001:
                break    
    return weights2


def SVM(train_set, test_set):
    X_train2 = train_set.values[0:, 0:260]
    Y_train2 = train_set.values[0:, 260:261]
    for i,val in enumerate(Y_train2):
        if Y_train2[i] == 0:
            Y_train2[i] = -1
    #print "X_train", type(X_train), X_train
    X_test2 = test_set.values[0:, 0:260]
    Y_test2 = test_set.values[0:, 260:261]
    for i,val in enumerate(Y_test2):
        if Y_test2[i] == 0:
            Y_test2[i] = -1
    #y_test = y_test.ravel()

    train_features2 = X_train2
    train_labels2 = Y_train2.reshape(Y_train2.shape[0],)
    test_label2 = Y_test2.reshape(Y_test2.shape[0],)

    #learning part
    #step_size = learning rate
    #print "X_train.shape", X_train.shape
    weights2 = linear_svm(train_features2, train_labels2, num_steps2 = 500, learning_rate2 = 0.5, add_intercept=False)
    final_scores2 = np.sign(np.dot(train_features2,weights2))
    accuracy_train2 = (final_scores2 == train_labels2).sum().astype(float) / len(final_scores2) 
    #print "accuracy_train svm", accuracy_train2
    #print 'Training Accuracy for svm: {0}'.format((np.sign(final_scores) == train_labels).sum().astype(float) / len(final_scores) * 100)

    # prediction for testing 

    final_scores22 = np.sign(np.dot(X_test2, weights2))
    accuracy_test2 = (final_scores22 == test_label2).sum().astype(float) / len(final_scores22) 
    #print "accuracy_test svm",accuracy_test2
    #print 'Testing Accuracy for svm: {0}'.format((final_scores2 == test_label).sum().astype(float) / len(final_scores2) * 100)

    return round(accuracy_train2,2), round(accuracy_test2,2)

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
    w1 = np.array(weights)
    for step in xrange(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        #print "output_error_signal",output_error_signal.shape
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


def LR (train_set, test_set):
    X_train = train_set.values[0:, 0:260]
    Y_train = train_set.values[0:, 260:261]

    X_test = test_set.values[0:, 0:260]
    Y_test = test_set.values[0:, 260:261]

    train_features = X_train
    train_labels = Y_train.reshape(Y_train.shape[0],)
    test_label = Y_test.reshape(Y_test.shape[0],)
    #learning part
    #step_size = learning rate
    weights = logistic_regression(train_features, train_labels, num_steps = 500, learning_rate = 0.01, add_intercept=True)
    #print "weights", weights.shape
    # prediction for training
    data_with_intercept = np.hstack((np.ones((train_features.shape[0], 1)),
                                     train_features))
    #print "data_with_intercept ",data_with_intercept.shape
    final_scores = np.dot(data_with_intercept, weights)
    preds = np.round(sigmoid(final_scores))
    accuracy_train = (preds == train_labels).sum().astype(float) / len(preds) 
    #print "return ", accuracy_train
    #print 'Training Accuracy for Logistic regression: {0}'.format((preds == train_labels).sum().astype(float) / len(preds) * 100)

    # prediction for testing 
    data_with_intercept2 = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    final_scores2 = np.dot(data_with_intercept2, weights)
    preds2 = np.round(sigmoid(final_scores2))
    accuracy_test = (preds2 == test_label).sum().astype(float) / len(preds2) 
    #print "return test", accuracy_test
    #print 'Testing Accuracy for Logistic regression: {0}'.format((preds2 == test_label).sum().astype(float) / len(preds2) * 100)

    return round(accuracy_train,2), round(accuracy_test,2)



def separate_dataset_instances(dataset):
    separated_instance = {}
    for i in range(len(dataset)):
        class_value = dataset[i]
        if (class_value[-1] not in separated_instance):
            separated_instance[class_value[-1]] = []
        separated_instance[class_value[-1]].append(class_value)
    return separated_instance


def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def attribute_summary(dataset):
    separated = separate_dataset_instances(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    denom1 = (2*math.pow(stdev,2))
    #print "denom1", denom1
    if denom1 == 0.0:
        denom1 = 0.00001
    exponent = math.exp(-(math.pow(x-mean,2)/(denom1))) 
    denom2 = (math.sqrt(2*math.pi) * stdev)
    if denom2 == 0.0:
        denom2 = 0.00001
    return (1 / denom2) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities
            
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) 

def NBC(train_set, test_set): #Naive Bayes 
    ############ Split train and test sets into feature & class lebels#########
    xtrain = train_set.values[0:, 0:261]
    xtest = test_set.values[0:, 0:261]
    # prepare model
    summaries = attribute_summary(xtrain) #ok
    # train model
    predictions_train = getPredictions(summaries, xtrain)
    accuracy_train = getAccuracy(xtrain, predictions_train)
    #print('Training Accuracy: {0}%').format(round(accuracy_train,2))


    # test model
    predictions_test = getPredictions(summaries, xtest)
    accuracy_test = getAccuracy(xtest, predictions_test)
    #print('Testing Accuracy: {0}%').format(round(accuracy_test,2))
    return round(accuracy_train,2), round(accuracy_test,2)

def union_trainset(j, t_frac, S1, S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10):
    train_set_union = []
    #print "j",j
    if j == 1:
        test_set = pd.DataFrame(S1)
        #print "S1", S1, type(S1), type(test_set) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 2:
        test_set = pd.DataFrame(S2)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 3:
        test_set = pd.DataFrame(S3)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2 , S4 , S5 , S6 , S7 , S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 4:
        test_set = pd.DataFrame(S4)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1, S2 , S3, S5 , S6 , S7 , S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 5:
        test_set = pd.DataFrame(S5)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2, S3 , S4, S6 , S7 , S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 6:
        test_set = pd.DataFrame(S6)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2 , S3, S4 , S5, S7 , S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 7:
        test_set = pd.DataFrame(S7)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1, S2 , S3 , S4 , S5 , S6, S8 , S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 8:
        test_set = pd.DataFrame(S8)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2, S3 , S4 , S5 , S6 , S7, S9 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 9:
        test_set = pd.DataFrame(S9)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2 , S3, S4 , S5 , S6 , S7 , S8 , S10), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    if j == 10:
        test_set = pd.DataFrame(S10)
        #print "S1", S1, type(S1) # array
        #train_set_union = np.array( [S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10])
        train_set_union = np.concatenate((S1 , S2 , S3, S4 , S5 , S6 , S7 , S8 , S9), axis=0) 
        #print "train_set_union",train_set_union, len(train_set_union), type (train_set_union), train_set_union.shape #ok
        train_set_union_df = pd.DataFrame(train_set_union) #convert Numpy ndarray to pandas dataframe
        #print "train_set_union_df", train_set_union_df, train_set_union_df.shape, type(train_set_union_df) #ok
        train_set = train_set_union_df.sample(frac=t_frac, random_state = 32)
    return train_set, test_set


t_frac = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
#t_frac = [0.2]
_lambda = 0.01 
trial = np.sqrt(10)

avg_accuracy_train_nbc = []
std_accuracy_train_nbc = []
sterr_train_nbc = []
avg_accuracy_test_nbc = []
std_accuracy_test_nbc = []
sterr_test_nbc = []


avg_accuracy_train_LR = []
std_accuracy_train_LR = []
sterr_train_LR = []
avg_accuracy_test_LR = []
std_accuracy_test_LR = []
sterr_test_LR = []


avg_accuracy_train_SVM = []
std_accuracy_train_SVM = []
sterr_train_SVM = []
avg_accuracy_test_SVM = []
std_accuracy_test_SVM = []
sterr_test_SVM = []

######## NBC ################
for i in range (0,len(t_frac)):
    accuracy_train_nbc_arr = []
    accuracy_test_nbc_arr = []
    for j in range (1,11):
        train_set, test_set = union_trainset(j, t_frac[i], S1, S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10)
        #print train_set.shape, type(train_set), test_set.shape, type(test_set) #ok column 261
        accuracy_train_nbc, accuracy_test_nbc = NBC(train_set, test_set)
        accuracy_train_nbc_arr.append(accuracy_train_nbc)
        accuracy_test_nbc_arr.append(accuracy_test_nbc)

    avg_accuracy_train_nbc.append((np.mean(accuracy_train_nbc_arr)))
    std_accuracy_train_nbc.append(np.std(accuracy_train_nbc_arr))
    avg_accuracy_test_nbc.append((np.mean(accuracy_test_nbc_arr)))
    std_accuracy_test_nbc.append(np.std(accuracy_test_nbc_arr))

for i in range (0,len(std_accuracy_train_nbc)):
    sterr_train_nbc.append(float(std_accuracy_train_nbc[i])/trial)
    sterr_test_nbc.append(float(std_accuracy_test_nbc[i])/trial)

############ LR #############
for i in range (0,len(t_frac)):
    accuracy_train_LR_arr = []
    accuracy_test_LR_arr = []

    for j in range (1,11):
        train_set, test_set = union_trainset(j, t_frac[i], S1, S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10)      
        accuracy_train_LR, accuracy_test_LR = LR(train_set, test_set)
        accuracy_train_LR_arr.append(accuracy_train_LR)
        accuracy_test_LR_arr.append(accuracy_test_LR)

    avg_accuracy_train_LR.append((np.mean(accuracy_train_LR_arr)))
    std_accuracy_train_LR.append(np.std(accuracy_train_LR_arr))
    avg_accuracy_test_LR.append((np.mean(accuracy_test_LR_arr)))
    std_accuracy_test_LR.append(np.std(accuracy_test_LR_arr))
   
for i in range (0,len(std_accuracy_train_LR)):
    sterr_train_LR.append(float(std_accuracy_train_LR[i])/trial)
    sterr_test_LR.append(float(std_accuracy_test_LR[i])/trial)

############ SVM #############
for i in range (0,len(t_frac)):

    accuracy_train_SVM_arr = []
    accuracy_test_SVM_arr = []
    
    for j in range (1,11):
        train_set, test_set = union_trainset(j, t_frac[i], S1, S2 , S3 , S4 , S5 , S6 , S7 , S8 , S9 , S10)
        #print train_set.shape, type(train_set), test_set.shape, type(test_set) #ok column 261
        accuracy_train_SVM, accuracy_test_SVM = SVM(train_set, test_set)
        #accuracy_train_SVM, accuracy_test_SVM = sklearn(train_set, test_set)
        accuracy_train_SVM_arr.append(accuracy_train_SVM)
        accuracy_test_SVM_arr.append(accuracy_test_SVM)
        

    avg_accuracy_train_SVM.append((np.mean(accuracy_train_SVM_arr)))
    std_accuracy_train_SVM.append((np.std(accuracy_train_SVM_arr)))
    avg_accuracy_test_SVM.append((np.mean(accuracy_test_SVM_arr)))
    std_accuracy_test_SVM.append((np.std(accuracy_test_SVM_arr)))
    
for i in range (0,len(std_accuracy_train_SVM)):
    sterr_train_SVM.append((float(std_accuracy_train_SVM[i])/trial))
    sterr_test_SVM.append((float(std_accuracy_test_SVM[i])/trial))

#print "avg_accuracy_train_nbc", avg_accuracy_train_nbc
#print "std_accuracy_train_nbc", std_accuracy_train_nbc
#print "sterr_train_nbc", sterr_train_nbc
print "avg_accuracy_test_nbc (validation accuracy)", avg_accuracy_test_nbc
print "std_accuracy_test_nbc", std_accuracy_test_nbc
print "sterr_test_nbc", sterr_test_nbc


#print "avg_accuracy_train_LR", avg_accuracy_train_LR
#print "std_accuracy_train_LR", std_accuracy_train_LR
#print "sterr_train_LR", sterr_train_LR
print "avg_accuracy_test_LR (validation accuracy)", avg_accuracy_test_LR
print "std_accuracy_test_LR", std_accuracy_test_LR
print "sterr_test_LR", sterr_test_LR


#print "avg_accuracy_train_SVM", avg_accuracy_train_SVM
##print "std_accuracy_train_SVM", std_accuracy_train_SVM
#print "sterr_train_SVM", sterr_train_SVM
print "avg_accuracy_test_SVM (validation accuracy)", avg_accuracy_test_SVM
print "std_accuracy_test_SVM", std_accuracy_test_SVM
print "sterr_test_SVM", sterr_test_SVM
#print train_set.columns


# create plot
size_S_c = 4680
size_train_data = []
for i in range (0,len(t_frac)):
    size_train_data.append(t_frac[i]*size_S_c)
#print size_train_data

#plt.plot( avg_accuracy_test_nbc, '-bo', label='Validation Accuracy NBC') 
#plt.plot( avg_accuracy_test_LR, '-g^', label='Validation Accuracy LR')
#plt.plot( avg_accuracy_test_SVM, '-r+', label='Validation Accuracy SVM')


plt.errorbar(size_train_data, avg_accuracy_test_nbc, yerr=sterr_test_nbc, marker='o', color='blue',ecolor='gray',elinewidth=1, capsize=2, barsabove = True, label='Validation Accuracy NBC')
plt.errorbar(size_train_data, avg_accuracy_test_LR, yerr=sterr_test_LR, marker='o', color='green',ecolor='purple',elinewidth=1, capsize=2, barsabove = True, label='Validation Accuracy LR')
plt.errorbar(size_train_data, avg_accuracy_test_SVM, yerr=sterr_test_SVM, marker='o', color='red',ecolor='black',elinewidth=1, capsize=2, barsabove = True, label='Validation Accuracy SVM')
plt.xlabel('Size of the training data')
plt.ylabel('Model accuracy')
plt.title('Learning curve')
#plt.xticks(index, ('0.01', '0.1', '0.2', '0.5', '0.6', '0.75', '0.9', '1'))
plt.xlim((100,1000))                 #Set X-axis limits
#plt.xticks(np.arange(len(size_train_data)), size_train_data[0:6])
#plt.errorbar(size_train_data, avg_accuracy_test_nbc, yerr=sterr_test_nbc, fmt='o', color='black',ecolor='lightgray', elinewidth=3, capsize=0);
plt.legend()
#plt.tight_layout()
plt.show()