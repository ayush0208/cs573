import pandas as pd
import numpy as np
from numpy.linalg import norm
import lr_svm as q2
import matplotlib.pyplot as plt
import scipy.stats

def get_kfold_split(input_filename, fold_number):
    train_set = pd.read_csv(input_filename)
    train_set = train_set.sample(random_state=18, frac=1)
    fold_number = 10
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
    new_X_train, new_y_train = q2.get_features_labels(new_train_set)
    new_X_test, new_y_test = q2.get_features_labels(test_set)
    return new_X_train, new_y_train, new_X_test, new_y_test

def get_train_test_data_nbc(fold_data_list, fraction, ind):
    test_set = fold_data_list[ind]
    rem_set = []
    for k, data in enumerate(fold_data_list):
        if(k!=ind):
            rem_set.append(fold_data_list[k])
    new_train_set = pd.concat(rem_set)
    new_train_set = new_train_set.sample(random_state=32, frac=fraction)
    return new_train_set, test_set 

def create_lookup_table(df, target_col, laplace_smoothing):
    lookup_table = {}
    value_counts = df[target_col].value_counts().sort_index()
    lookup_table['class_name'] = value_counts.index.to_numpy()
    lookup_table['class_count'] = value_counts.values

    data_columns = df.drop(target_col, axis=1).columns
    for col in data_columns:
        lookup_table[col] = {}

        counts = df.groupby(target_col)[col].value_counts()
        df_counts = counts.unstack(target_col)
        if laplace_smoothing:
            if df_counts.isna().any(axis=None):
                df_counts.fillna(value=0, inplace = True)
                df_counts+=1
        df_probabilities = df_counts/df_counts.sum()
        for val in df_probabilities.index:
            probability = df_probabilities.loc[val].to_numpy()
            lookup_table[col][val] = probability
    return lookup_table

def predict(row, result):
    class_estimates = result['class_count']
    row = row[:-1] 
    for feature in row.index:
        try:
            value = row[feature]
            prob = result[feature][value]
            class_estimates = class_estimates * prob
        except KeyError:
            continue
    max_class_index = class_estimates.argmax()
    prediction = result['class_name'][max_class_index]
    return prediction

def accuracy(lookup_table, df, target_col):
    predictions = df.apply(predict, axis=1, args=(lookup_table,))
    correct_predictions = predictions == df[target_col]
    return correct_predictions.mean()

def compare_models(fold_data_list_nbc, fold_data_list, fraction_list, fold_number):
    svm_model_accuracy = []
    lr_model_accuracy = []
    nbc_model_accuracy = []
    for frac in fraction_list:
        svm_frac_accuracy = []
        lr_frac_accuracy = []
        nbc_frac_accuracy = []
        for ind in range(fold_number):
            new_X_train, new_y_train, new_X_test, new_y_test = get_train_test_data(fold_data_list, frac, ind)

            # train lr
            weights = q2.logistic_regression(new_X_train, new_y_train, 0.01, 500, 0.01, True) 
            lr_frac_accuracy.append(q2.lr_accuracy(new_X_test, new_y_test, weights))

            # train svm
            weights = q2.run_svm(new_X_train, new_y_train, 0.01, 500, 0.5, True)
            svm_frac_accuracy.append(q2.model_accuracy(new_X_test, new_y_test, weights))

            #dataset for NBC
            train_df, test_df = get_train_test_data_nbc(fold_data_list_nbc, frac, ind)
            target_col = 'decision'
            
            #train NBC
            prob_table = create_lookup_table(train_df, target_col, True)
            nbc_frac_accuracy.append(round(accuracy(prob_table, test_df, target_col), 2))

        svm_mean = np.mean(svm_frac_accuracy)
        svm_std = np.sqrt(np.var(svm_frac_accuracy))
        svm_std_err = svm_std/np.sqrt(10)
        svm_model_accuracy.append([frac, svm_mean, svm_std_err])

        lr_mean = np.mean(lr_frac_accuracy)
        lr_std = np.sqrt(np.var(lr_frac_accuracy))
        lr_std_err = lr_std/np.sqrt(10)
        lr_model_accuracy.append([frac, lr_mean, lr_std_err])

        nbc_mean = np.mean(nbc_frac_accuracy)
        nbc_std = np.sqrt(np.var(nbc_frac_accuracy))
        nbc_std_err = nbc_std/np.sqrt(10)
        nbc_model_accuracy.append([frac, nbc_mean, nbc_std_err])

    svm_data = np.array(svm_model_accuracy)
    lr_data = np.array(lr_model_accuracy)
    nbc_data = np.array(nbc_model_accuracy)
    dataset_size = [element*4680 for element in fraction_list]
    plt.errorbar(dataset_size, lr_data[:, 1], yerr=lr_data[:, 2], label='LR', marker = 'o')
    plt.errorbar(dataset_size, svm_data[:, 1], yerr=svm_data[:, 2], label='SVM', marker = 'o')
    plt.errorbar(dataset_size, nbc_data[:, 1], yerr=nbc_data[:, 2], label='NBC', marker = 'o')
    plt.xlabel('Training Set size')
    plt.ylabel('Model Accuracy')
    plt.title('Performance of models')
    plt.legend()
    plt.show()
    return svm_model_accuracy, lr_model_accuracy, nbc_model_accuracy

def calculate_p_value(svm_data, lr_data):
    p = scipy.stats.ttest_rel(lr_data, svm_data).pvalue
    print("Value of p:" , p)
    if p<0.01:
        return "Accepting alternative hypothesis and rejecting null hypothesis"
    else:
        return "Accepting null hypothesis"

def main_cv():
    fold_number = 10
    fold_data_list = get_kfold_split('./trainingSet.csv', fold_number)
    fold_data_list_nbc = get_kfold_split('./trainingSet_NBC.csv', fold_number)
    fraction_list = [0.025,0.05,0.075,0.1,0.15,0.2]
    svm_data, lr_data, nbc_data = compare_models(fold_data_list_nbc, fold_data_list, fraction_list, fold_number)

    # hypothesis testing
     
    # lr_data = [element[1] for element in lr_data]
    # svm_data = [element[1] for element in svm_data]
    # hypothesis = calculate_p_value(svm_data, lr_data)
    # print(hypothesis)


if __name__ == "__main__":
    main_cv()
