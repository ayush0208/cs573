from matplotlib import markers
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def effect_of_fraction(input_filename, fraction_list):
    training_accuracy_list = []
    testing_accuracy_list = []
    for frac in fraction_list:
        print("Frac", frac)
        df = pd.read_csv(input_filename)
        train_df, test_df = split.train_test_split(df)
        train_df.to_csv('./trainingSet.csv', index = False)
        test_df.to_csv('./testSet.csv', index = False)
        resultant_table, train_df, test_df = naiveBayes.nbc(frac)
        target_col = 'decision'
        training_accuracy_list.append(naiveBayes.accuracy(resultant_table, train_df, target_col))
        testing_accuracy_list.append(naiveBayes.accuracy(resultant_table, test_df, target_col))
        print("Training Accuracy: ",round(naiveBayes.accuracy(resultant_table, train_df, target_col), 2))
        print("Testing Accuracy: ", round(naiveBayes.accuracy(resultant_table, test_df, target_col), 2))
    return training_accuracy_list, testing_accuracy_list

def plot_graph(fraction_list, training_accuracy_list, testing_accuracy_list):
    plt.plot(fraction_list, training_accuracy_list, label = "Training Accuracy", marker = 'o')
    plt.plot(fraction_list, testing_accuracy_list, label = "Testing Accuracy", marker = 'o')
    
    plt.xlabel('Fraction size')
    plt.ylabel('Accuracy')
    plt.title('Comparing accuracy with t_frac')
    plt.legend()
    plt.show()

def t_frac_effect_accuracy(input_filename, fraction_list):
    training_accuracy_list, testing_accuracy_list = effect_of_fraction(input_filename, fraction_list)
    plot_graph(fraction_list, training_accuracy_list, testing_accuracy_list)

if __name__ == '__main__':

    naiveBayes = __import__('5_1')
    import discretize as binning
    import split as split

    fraction_list = [0.01,0.1,0.2,0.5,0.6,0.75,0.9,1]
    input_filename = './dating-binned.csv'
    t_frac_effect_accuracy(input_filename, fraction_list)
    