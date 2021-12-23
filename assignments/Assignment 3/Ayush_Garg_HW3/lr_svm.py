import pandas as pd
import numpy as np
from numpy.linalg import norm
import sys
import warnings
warnings.filterwarnings("ignore")

# LR
def sigmoid(z):
	    return(1 / (1 + np.exp(-z))) 

def cost_func(features, target, weights, lamb):
    scores = np.dot(features, weights)
    cf = np.sum( target*scores - np.log(1 + np.exp(scores)) )  +  lamb/(2) * np.dot(weights, weights)
    return cf

def logistic_regression(features, target, lamb, num_steps, learning_rate, add_intercept):
    if add_intercept:
        bias = np.ones((features.shape[0], 1))
        features = np.concatenate((bias, features), axis=1)
    
    weights = np.zeros(features.shape[1]) 
    
    for step in range(0, num_steps):
        w1 = np.array(weights)
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
       	output_error_signal = predictions - target
        gradient = np.dot(features.T, output_error_signal) +  (lamb * weights)
        weights -= learning_rate * gradient
        w2 = np.array(weights)
        diff = np.subtract(w2, w1)
        l2 = norm(diff)
        if l2 < 0.000001:
        	break
    return weights

def lr_accuracy(X, y, weights):
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)
    preds = np.round(sigmoid(np.dot(X, weights)))
    accuracy = (preds == y).sum().astype(float) / len(preds)
    return round(accuracy, 2)

def print_accuracy(train_features, train_labels, X_test, test_label, weights):
    print ('Training Accuracy LR:', lr_accuracy(train_features, train_labels, weights))
    print ('Testing Accuracy LR:', lr_accuracy(X_test, test_label, weights))

# SVM
def loss_function(X, y, weights, lamb):
    N = X.shape[0]
    prod = 1 - y * (np.dot(X, weights))
    prod[prod < 0] = 0 
    hinge_loss =  (np.sum(prod) / N)

    loss = lamb / 2 * np.dot(weights, weights) + hinge_loss
    return loss

def gradient_calculation(X, y, weights, lamb):
    distance = 1 - y * (np.dot(X, weights))
    dw = np.zeros(len(weights))
    t = np.where(y <= 0, -1, 1)
    for ind, d in enumerate(distance):
        condition = t[ind] * (np.dot(X[ind], weights)) >= 1
        if condition:
            dw += lamb * weights
        else:
            dw += lamb * weights - np.dot(X[ind], t[ind])
    return dw/len(y)

def run_svm(X, y, lamb, max_iter, step_size, add_intercept):
    if(add_intercept):
        bias = np.ones((X.shape[0], 1))
        X = np.concatenate((bias, X), axis=1)
    weights = np.zeros(X.shape[1])
    for epoch in range(max_iter):
        dw = gradient_calculation(X, y, weights, lamb)
        diff =  step_size * dw
        weights -= diff
        if(norm(diff)< 0.000001):
            # print("Tolerance reached")
            break
    return weights        

def predict(X, weights):
    return np.sign(np.dot(X, weights))

def model_accuracy(X, y, weights):
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)
    preds = predict(X, weights)
    t = np.where(y <= 0, -1, 1)
    accuracy = (preds == t).sum()/t.size
    return accuracy

def get_accuracy(X_train, y_train, X_test, y_test, weights):
    print("Training Accuracy SVM:" , round(model_accuracy(X_train, y_train, weights), 2))
    print("Testing Accuracy SVM:" , round(model_accuracy(X_test, y_test, weights), 2))

def get_features_labels(df):
    labels = df['decision']
    train_features = df.drop('decision', 1)
    return train_features.to_numpy(), labels.to_numpy()



if __name__ == "__main__":
    modelIdx = sys.argv[3]
    train_data = pd.read_csv(sys.argv[1])
    X_train, y_train = get_features_labels(train_data)
    test_data = pd.read_csv(sys.argv[2])
    X_test, y_test = get_features_labels(test_data)

    if(modelIdx == '1'):
        weights = logistic_regression(X_train, y_train, lamb=0.01, num_steps = 500, learning_rate = 0.01, add_intercept=True)
        print_accuracy(X_train, y_train, X_test, y_test, weights)

    if(modelIdx == '2'):
        weights = run_svm(X_train, y_train, 0.01, 500, 0.5, True)
        get_accuracy(X_train, y_train, X_test, y_test, weights)