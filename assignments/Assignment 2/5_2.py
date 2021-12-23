import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def effect_of_bins(input_filename, bin_value_list):
    training_accuracy_list = []
    testing_accuracy_list = []
    # bin_value_list = [2,5,10,50,100,200]
    for bin in bin_value_list:
        print("Bin size:", bin)
        df = pd.read_csv(input_filename)
        binned_dict = binning.binning_continuous_valued_columns(df, continuous_valued_columns, preference_scores_of_participant, preference_scores_of_partner, bin)
        train_df, test_df = split.train_test_split(df)
        train_df.to_csv('./trainingSet.csv', index = False)
        test_df.to_csv('./testSet.csv', index = False)
        resultant_table, train_df, test_df = naiveBayes.nbc(1)
        target_col = 'decision'
        training_accuracy_list.append(naiveBayes.accuracy(resultant_table, train_df, target_col))
        testing_accuracy_list.append(naiveBayes.accuracy(resultant_table, test_df, target_col))
        print("Training Accuracy:",round(naiveBayes.accuracy(resultant_table, train_df, target_col), 2))
        # print("Testing Accuracy: ", naiveBayes.accuracy(resultant_table, test_df, target_col))
        print("Testing Accuracy:", round(naiveBayes.accuracy(resultant_table, test_df, target_col), 2))
    return training_accuracy_list, testing_accuracy_list

def plot_graph(bin_value_list, training_accuracy_list, testing_accuracy_list):
    plt.plot(bin_value_list, training_accuracy_list, label = "Training Accuracy", marker = 'o')
    plt.plot(bin_value_list, testing_accuracy_list, label = "Testing Accuracy", marker = 'o')
    
    plt.xlabel('Bin size')
    plt.ylabel('Accuracy')
    plt.title('Comparing accuracy with bin size')
    plt.legend()
    plt.show()

def bin_effect(input_filename, bin_value_list):
    training_accuracy_list, testing_accuracy_list = effect_of_bins(input_filename, bin_value_list)
    plot_graph(bin_value_list, training_accuracy_list, testing_accuracy_list)

if __name__ == '__main__':

    # import 5_1 as naiveBayes
    naiveBayes = __import__('5_1')
    import discretize as binning
    import split as split

    preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']
    preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']
    continuous_valued_columns = ['age', 'age_o', 'importance_same_race', 'importance_same_religion', 
        'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
       'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests',
       'attractive_important', 'sincere_important', 'intelligence_important',
       'funny_important', 'ambition_important', 'shared_interests_important',
       'attractive', 'sincere', 'intelligence', 'funny', 'ambition',
       'attractive_partner', 'sincere_partner', 'intelligence_parter',
       'funny_partner', 'ambition_partner', 'shared_interests_partner',
       'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking',
       'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts',
       'music', 'shopping', 'yoga', 'interests_correlate',
       'expected_happy_with_sd_people', 'like']
    
    bin_value_list = [2,5,10,50,100,200]
    input_filename = './dating.csv'
    bin_effect(input_filename, bin_value_list)