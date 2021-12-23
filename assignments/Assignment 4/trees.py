import pandas as pd
import numpy as np
import sys
import random
from random import randrange
import multiprocessing
from multiprocessing import Pool
from itertools import repeat

def majority_vote(data, target_label):
    decision_column=data[target_label]
    max_count=0
    max_key=-100
    decision_count = decision_column.value_counts()
    for key in decision_count.keys():
        value=decision_count[key]
        if value>max_count:
            max_count=value
            max_key=key
    return max_key

def gini(data, feature):
    study_column=data[feature]
    feature_count=study_column.value_counts()
    total_gini=1
    for key in feature_count.keys():
        value=feature_count[key]
        total_gini-= pow((value/len(study_column)),2)
    return total_gini

def best_feature_split(data, feature_list, target_label):
    choose_feature=None
    best_gain=-1000000
    total_gini = gini(data, target_label)

    for feature in feature_list[:-1]:
        study_column=data[feature]
        feature_count=study_column.value_counts()
        feature_gain=total_gini
        for key in feature_count.keys():
            value=feature_count[key]
            percentage=value/len(study_column)
            study_data=data[data[feature]==key]
            sub_gini = gini(study_data, target_label)
            feature_gain-=percentage*sub_gini
        if feature_gain>best_gain:
            best_gain=feature_gain
            choose_feature=feature
    return choose_feature, best_gain

def build_tree(tree_dict, data, feature_list, target_label, max_depth, min_example):
    if len(feature_list)==1:
        return majority_vote(data, target_label)

    choose_feature, _ = best_feature_split(data, feature_list, target_label)
    tmp_dict={}
    feature_count = data[choose_feature].value_counts()
    use_feature=feature_list.copy()
    use_feature.remove(choose_feature)

    if  len(data)<= min_example or max_depth == 0:
        tree_dict[choose_feature] = majority_vote(data[data[choose_feature]], target_label)
        return tree_dict

    tmp_dict['major']=majority_vote(data, target_label)
    for key in feature_count.keys():
        value=feature_count[key]
        if value <= min_example or max_depth == 1:
            tmp_dict[key] = majority_vote(data[data[choose_feature]==key], target_label)
        else:
            tmp_dict[key]=build_tree({},data[data[choose_feature]==key],use_feature, target_label, max_depth-1,min_example)
            
    tree_dict[choose_feature]=tmp_dict
    return tree_dict

def build_tree_rf(tree_dict, data, feature_list, target_label, max_depth, min_example):
    if len(feature_list)==1:
        return majority_vote(data, target_label)

    red_feature_list=random.sample(feature_list[:-1], int(np.sqrt(len(feature_list)-1)))+[feature_list[-1]]
    choose_feature, _ = best_feature_split(data, red_feature_list, target_label)
    tmp_dict={}
    feature_count = data[choose_feature].value_counts()
    use_feature=feature_list.copy()
    use_feature.remove(choose_feature)

    if  len(data)<= min_example or max_depth == 0:
        tree_dict[choose_feature] = majority_vote(data[data[choose_feature]], target_label)
        return tree_dict

    tmp_dict['major']=majority_vote(data, target_label)
    for key in feature_count.keys():
        value=feature_count[key]
        if value <= min_example or max_depth == 1:
            tmp_dict[key] = majority_vote(data[data[choose_feature]==key], target_label)
        else:
            tmp_dict[key]=build_tree_rf({},data[data[choose_feature]==key],use_feature, target_label, max_depth-1,min_example)
            
    tree_dict[choose_feature]=tmp_dict
    return tree_dict

def pred_label(tree_dict,data):
    for key in tree_dict.keys():
        new_dict=tree_dict[key]
        value=data[key]
        if type(new_dict)!=dict:
            return new_dict
        if value not in new_dict:
            return new_dict['major']

        now_dict=new_dict[value]
        if type(now_dict)!=dict:
            return now_dict
        else:
            return pred_label(now_dict,data)

def pred(tree_dict,data):
    num = len(data)
    count=0
    for i in range(len(data)):
        pred=pred_label(tree_dict,data.iloc[i,:])
        if pred==data.iloc[i,-1]:
            count+=1
    return float(count)/num

def decisionTree(trainingSet, testSet, max_depth):
    min_example=50
    
    tree_dict={}
    feature_list=list(trainingSet.columns)
    target_label = feature_list[-1]

    tree_dict=build_tree(tree_dict,trainingSet,feature_list, target_label, max_depth,min_example)
    train_accu=pred(tree_dict,trainingSet)
    test_accu=pred(tree_dict,testSet)
    return train_accu,test_accu

def DT(trainingSet, max_depth, min_example, modelIdx=1):
    tree_dict={}
    feature_list=list(trainingSet.columns)
    target_label = feature_list[-1]
    if(modelIdx==1):
        tree_dict=build_tree(tree_dict,trainingSet,feature_list,target_label,max_depth,min_example)
    else:
        tree_dict = build_tree_rf(tree_dict,trainingSet,feature_list,target_label,max_depth,min_example)
    return tree_dict

def Evaluate_BT(Tree_lists, data):
    count=0
    for i in range(len(data)):
        pred_list={}
        for k in range(len(Tree_lists)):
            pred=pred_label(Tree_lists[k],data.iloc[i,:])
            if pred not in pred_list:
                pred_list[pred]=1
            else:
                pred_list[pred] += 1
        max_key=-1
        max_count=-1000
        for key in pred_list.keys():
            if pred_list[key]>max_count:
                max_count=pred_list[key]
                max_key=key
        if max_key==data.iloc[i,-1]:
            count+=1
    return count/len(data)

def bagging(pool, trainingSet,testSet, max_depth, numTrees):
    Tree_lists = []
    percent = 1
    random.seed(0)
    tmp_train = []
    for k in range(numTrees):
        rand_state=random.randint(0,1000000)
        tmp_train_ind = trainingSet.sample(random_state=rand_state, replace=True,frac=percent)
        tmp_train.append(tmp_train_ind)
    args = list(zip(tmp_train, repeat(max_depth), repeat(50)))
    Tree_lists = pool.starmap(DT, args)
    train_accu=Evaluate_BT(Tree_lists,trainingSet)
    test_accu = Evaluate_BT(Tree_lists, testSet)
    return train_accu,test_accu

def randomForests(pool, trainingSet,testSet, max_depth, numTrees):
    Tree_lists = []
    percent = 0.5
    tmp_train = []
    random.seed(0)
    for k in range(numTrees):
        rand_state = random.randint(0, 1000000)
        tmp_train_ind = trainingSet.sample(random_state=rand_state, replace=True,frac=percent)
        tmp_train.append(tmp_train_ind)
    args = list(zip(tmp_train, repeat(max_depth), repeat(50), repeat(3)))
    Tree_lists = pool.starmap(DT, args)
    train_accu = Evaluate_BT(Tree_lists, trainingSet)
    test_accu = Evaluate_BT(Tree_lists, testSet)
    return train_accu, test_accu

def print_accuracy(train_acc, test_acc, model):
    if(model=='1'):
        print('Training Accuracy DT: %.2f'%train_acc)
        print('Testing Accuracy DT: %.2f' % test_acc)
    elif(model=='2'):
        print('Training Accuracy BT: %.2f'%train_acc)
        print('Testing Accuracy BT: %.2f' % test_acc)
    elif(model=='3'):
        print('Training Accuracy RF: %.2f'%train_acc)
        print('Testing Accuracy RF: %.2f' % test_acc)

def models(trainingDataFilename, testDataFilename, modelIdx, pool):
    train_df = pd.read_csv(trainingDataFilename)
    test_df = pd.read_csv(testDataFilename)

    if (modelIdx == '1'):
        depth = 8
        train_acc, test_acc = decisionTree(train_df, test_df, depth)
        print_accuracy(train_acc, test_acc, modelIdx)
    
    elif (modelIdx == '2'):
        train_acc,test_acc=bagging(pool, train_df,test_df, 8, 30)
        print_accuracy(train_acc, test_acc, modelIdx)

    elif (modelIdx == '3'):
        train_acc, test_acc = randomForests(pool, train_df, test_df, 8, 30)
        print_accuracy(train_acc, test_acc, modelIdx)

if __name__ == "__main__":
    pool = Pool(processes=6)
    models(sys.argv[1], sys.argv[2], sys.argv[3], pool)
    pool.close()