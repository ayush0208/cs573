import pandas as pd
import numpy as np
import sys

def label_encoding(df, label_encoding_list):
    encoding_dict = {}
    for col in label_encoding_list:
        unique_val = np.unique(df[col].values)
        val_dict = {}
        encoding_value = 0
        for val in unique_val:
            val_dict[val]=encoding_value
            df[col].replace(val, encoding_value, inplace=True)
            encoding_value+=1
        encoding_dict[col] = val_dict
    return df

def normalize(df, preference_scores_of_participant, preference_scores_of_partner):
    for ind in df.index:
        total=0
        for col in preference_scores_of_participant:
            total+= df[col][ind]
        for col in preference_scores_of_participant:
            df.loc[ind, col] = df[col][ind]/total
        total=0
        for col in preference_scores_of_partner:
            total+= df[col][ind]
        for col in preference_scores_of_partner:
            df.loc[ind, col] = df[col][ind]/total
    return df

def binning_continuous_valued_columns(df, continuous_valued_columns, preference_scores_of_participant, preference_scores_of_partner, number_of_bins):
    binned_dict = {}
    for col in continuous_valued_columns:
        min_val = 0
        max_val = 10
        if col=='age' or col == 'age_o':
            min_val = 18
            max_val = 58
        elif col in preference_scores_of_participant or col in preference_scores_of_partner:
            min_val = 0
            max_val = 1
        elif col == 'interests_correlate':
            min_val = -1
            max_val = 1
        bins = np.linspace(min_val,max_val,number_of_bins+1)
        df.loc[df[col] < min_val, col] = max_val
        df.loc[df[col] > max_val, col] = max_val
        df[col] = pd.cut(df[col],bins, include_lowest=True, labels= np.arange(number_of_bins))
        binned_dict[col] = df[col].value_counts().sort_index().values
    return binned_dict

def train_test_split(df):
    test_df = df.sample(random_state=47, frac=0.2)
    train_df = df.drop(test_df.index)
    return train_df, test_df

def preprocessing():
    num_of_rows = 6500
    df = pd.read_csv('./dating-full.csv', nrows=num_of_rows)

    # for 1-a
    df.drop(columns = ['race_o', 'race', 'field'], axis=1, inplace=True)

    # for 1-b
    label_encoding_list = ['gender']
    df = label_encoding(df, label_encoding_list)

    # for 1-c
    preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']
    preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']
    df = normalize(df, preference_scores_of_participant, preference_scores_of_partner)

    # for 1-d
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

    binned_dict = binning_continuous_valued_columns(df, continuous_valued_columns, preference_scores_of_participant, preference_scores_of_partner,2)

    # for 1-e
    train_df, test_df = train_test_split(df)
    train_df.to_csv('trainingSet.csv', index = False)
    test_df.to_csv('testSet.csv', index = False)

if __name__ == "__main__":
    preprocessing()