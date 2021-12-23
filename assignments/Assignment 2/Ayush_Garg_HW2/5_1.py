import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

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
    row = row[:-1] # removing the 
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

def nbc(t_frac):
    df = pd.read_csv('./trainingSet.csv')
    train_df = df.sample(random_state=47, frac=t_frac)
    target_col = 'decision'
    test_df = pd.read_csv('./testSet.csv')
    prob_table = create_lookup_table(train_df, target_col, True)
    return prob_table, train_df, test_df

def naive_bayes_classifier():
    prob_table, train_df, test_df = nbc(1)
    target_col = 'decision'
    print("Training Accuracy: ", round(accuracy(prob_table, train_df, target_col), 2))
    print("Testing Accuracy: ", round(accuracy(prob_table, test_df, target_col), 2))

if __name__ == "__main__":
    naive_bayes_classifier()