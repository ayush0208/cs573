import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

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
        # print(col, df[col].value_counts().sort_index().values)
    return binned_dict

def discretize_continuous_attributes(input_filename, output_filename):
    df = pd.read_csv(input_filename)

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

    binned_dict = binning_continuous_valued_columns(df, continuous_valued_columns, preference_scores_of_participant, preference_scores_of_partner,5)
    for item in binned_dict:
        print("{}: {}".format(item, binned_dict[item]))
    df.to_csv(output_filename, index = False)

if __name__ == "__main__":
    discretize_continuous_attributes(sys.argv[1], sys.argv[2])