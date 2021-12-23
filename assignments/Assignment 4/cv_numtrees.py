import pandas as pd
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import trees as q2
from multiprocessing import Pool
from itertools import repeat
import scipy.stats

def get_kfold_split(train_set, fold_number):
    fold_size=len(train_set)/fold_number
    fold_data_list=[]
    for i in range(fold_number):
        new_fold=train_set.iloc[int(i*fold_size):int((i+1)*fold_size),:]
        fold_data_list.append(new_fold)
    return fold_data_list

def get_train_test_data(fold_data_list, fraction, ind):
    test_set = fold_data_list[ind]
    rem_set = []
    for k, data in enumerate(fold_data_list):
        if(k!=ind):
            rem_set.append(fold_data_list[k])
    new_train_set = pd.concat(rem_set)
    new_train_set = new_train_set.sample(random_state=32, frac=fraction)
    return new_train_set, test_set

def compare_models(pool, fold_data_list, tree_list, fold_number):
    bt_model_accuracy = []
    rf_model_accuracy = []
    train_set_list = []
    test_set_list = []
    for ind in range(fold_number):
        train_set_ind, test_set_ind = get_train_test_data(fold_data_list, 1, ind)
        train_set_list.append(train_set_ind)
        test_set_list.append(test_set_ind)
    for t in tree_list:
        print("No of trees:", t)
        bt_frac_accuracy = []
        rf_frac_accuracy = []
        for ind in range(fold_number):
            print("Current fold:", ind)

            # train bt
            _ ,test_acc = q2.bagging(pool, train_set_list[ind], test_set_list[ind], 8, t) 
            print("Bagging acc", test_acc)
            bt_frac_accuracy.append(test_acc)
            
            #train rf
            _ ,test_acc = q2.randomForests(pool, train_set_list[ind], test_set_list[ind], 8, t) 
            print("Random Forest acc", test_acc)
            rf_frac_accuracy.append(test_acc)

        bt_mean = np.mean(bt_frac_accuracy)
        bt_std = np.sqrt(np.var(bt_frac_accuracy))
        bt_std_err = bt_std/np.sqrt(fold_number)
        bt_model_accuracy.append([t, bt_mean, bt_std_err])

        rf_mean = np.mean(rf_frac_accuracy)
        rf_std = np.sqrt(np.var(rf_frac_accuracy))
        rf_std_err = rf_std/np.sqrt(fold_number)
        rf_model_accuracy.append([t, rf_mean, rf_std_err])

        print("Bagging:{}, Random Forest:{}".format(bt_mean, rf_mean))

    bt_data = np.array(bt_model_accuracy)
    rf_data = np.array(rf_model_accuracy)
    plt.errorbar(tree_list, bt_data[:, 1], yerr=bt_data[:, 2], label='Bagging', marker = 'o')
    plt.errorbar(tree_list, rf_data[:, 1], yerr=rf_data[:, 2], label='Random Forest', marker = 'o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Model Accuracy')
    plt.title('Performance of models')
    plt.legend()
    plt.show()
    return bt_model_accuracy, rf_model_accuracy

def calculate_p_value(svm_data, lr_data):
    p = scipy.stats.ttest_rel(lr_data, svm_data).pvalue
    print("Value of p:" , p)
    if p<0.01:
        return "Accepting alternative hypothesis and rejecting null hypothesis"
    else:
        return "Accepting null hypothesis"

if __name__ == "__main__":
    pool = Pool(processes=6)
    train_df = pd.read_csv('trainingSet.csv')
    train_df = train_df.sample(random_state=18, frac=1)
    percent = 0.5
    half_train_df = train_df.sample(random_state=32, frac=percent)
    len(half_train_df)
    fold_number = 10
    fold_data_list = get_kfold_split(half_train_df, fold_number)
    tree_list = [10, 20, 40, 50]
    bt_model_accuracy_trees, rf_model_accuracy_trees = compare_models(pool, fold_data_list, tree_list, fold_number)
    pool.close()

    # hypothesis testing
     
    # rf_model_accuracy = [element[1] for element in rf_model_accuracy_trees]
    # bt_model_accuracy = [element[1] for element in bt_model_accuracy_trees]
    # hypothesis = calculate_p_value(rf_model_accuracy, bt_model_accuracy)
    # print(hypothesis)