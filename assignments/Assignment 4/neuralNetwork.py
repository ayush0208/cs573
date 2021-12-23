import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def get_accuracy(label,pred,threshold):
    count=0
    for i in range(len(label)):
        if pred[i]<threshold and label[i]==0:
            count+=1
        elif pred[i]>threshold and label[i]==1:
            count+=1
    return count/float(len(label))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

def predict(x, weights):
    x = np.array(x)
    temp = np.ones(x.shape[0] + 1)
    temp[0:-1] = x
    a = temp
    for l in range(0, len(weights)):
        a = sigmoid(np.dot(a, weights[l]))
    return a

def pred_array(array, weights):
    pred=[]
    for i in range(len(array)):
        result=predict(array[i], weights)
        pred.append(result)
    return pred

def fit(X, y, weights, learning_rate, epochs):
    X = np.atleast_2d(X)
    temp = np.ones([X.shape[0], X.shape[1] + 1])
    temp[:, 0:-1] = X
    X = temp
    y = np.array(y)
    update_weight=[0]*len(weights)
    for k in range(epochs):
        for i in range(len(X)):
            example=X[i]
            pred=[example]

            for l in range(len(weights)):
                pred.append(sigmoid(np.dot(pred[l],weights[l])))

            error=y[i]-pred[-1]
            deltas=[error*sigmoid_deriv(pred[-1])]
            for l in range(len(pred) - 2, 0, -1):
                deltas.append(deltas[-1].dot(weights[l].T) * sigmoid_deriv(pred[l]))
            deltas.reverse()
            for l in range(len(weights)):
                layer = np.atleast_2d(pred[l])
                delta = np.atleast_2d(deltas[l])
                update_weight[l] += learning_rate * layer.T.dot(delta)
        for l in range(len(weights)):
            weights[l]+=update_weight[l]
    return weights

def neural_net(layers, train_set):
    weights = []
    for i in range(1, len(layers)-1):
        weights.append(np.random.random((layers[i - 1] + 1, layers[i] + 1))-0.5)
    weights.append(np.random.random((layers[len(layers) - 2] + 1, layers[len(layers)-1])) - 0.5)

    weights = fit(np.array(train_set.iloc[:,:-1]), np.array(train_set.iloc[:,-1]), weights, 0.001,50)
    return weights

def get_model_accuracy(train_set, test_set, weights, threshold):
    train_pred = pred_array(np.array(train_set.iloc[:,:-1]), weights)
    train_accu=get_accuracy(np.array(train_set.iloc[:,-1]),train_pred,threshold)
    print('Training Accuracy NN: %.2f' % train_accu)

    test_pred = pred_array(np.array(test_set.iloc[:,:-1]), weights)
    test_accu=get_accuracy(np.array(test_set.iloc[:,-1]),test_pred,threshold)
    print('Testing Accuracy NN: %.2f' % test_accu)

if __name__ == "__main__":
    train_reader = pd.read_csv('trainingSet.csv')
    test_reader = pd.read_csv('testSet.csv')
    activation = 'sigmoid'
    threshold = 0.5
    features=len(train_reader.columns)-1
    weights = neural_net([features,50,20,1], train_reader)
    get_model_accuracy(train_reader, test_reader, weights, threshold)