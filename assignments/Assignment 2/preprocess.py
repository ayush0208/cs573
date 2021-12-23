import numpy as np
import pandas as pd
import sys


# for 1-a problem
def contains_single_quote(s):
    if s.startswith("'") or s.endswith("'"):
        return True
    else:
        return False

def strip_quotes(df, strip_quotes_list):
    count = 0
    for index in df.index:
        for col in strip_quotes_list:
            if contains_single_quote(df.loc[index, col]):
                count+=1
                df.loc[index, col] = df.loc[index, col].replace('\'','')
    return count

# for 1-b problem
def to_lower(df):
    val = df['field'].values
    count=0
    for value in val:
        if not(value.islower()):
            count+=1 
    df['field'] = df['field'].str.lower()
    return df, count

# for 1-c problem
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
    print("Value assigned for male in column gender:", encoding_dict['gender']['male'])
    print("Value assigned for European/Caucasian-American in column race:", encoding_dict['race']['European/Caucasian-American'])
    print("Value assigned for Latino/Hispanic American in column race_o:", encoding_dict['race_o']['Latino/Hispanic American'])
    print("Value assigned for law in column field:", encoding_dict['field']['law'])
    return df

# for 1-d problem
rating_of_partner_from_participant = ['attractive_partner', 'sincere_partner', 'intelligence_parter','funny_partner', 'ambition_partner', 'shared_interests_partner']
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
       'expected_happy_with_sd_people', 'like'
    ]
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
            # df[col][ind] = df[col][ind]/total
    for col in preference_scores_of_participant:
        print("Mean of "+ col + ": " ,round(df[[col]].mean()[0], 2))
    for col in preference_scores_of_partner:
        print("Mean of "+ col + ": " , round(df[[col]].mean()[0], 2))
    return df    


def preprocessing(input_filename, output_filename):
    df = pd.read_csv(input_filename)
    # for 1-a
    strip_quotes_list = ['race','race_o','field']
    count = strip_quotes(df, strip_quotes_list)
    print("Quotes removed from " + str(count) + " cells")

    # for 1-b
    df, count = to_lower(df)
    print("Standardized " + str(count) + " cells to lower case")

    # for 1-c
    label_encoding_list = ['gender','race','race_o','field']
    df = label_encoding(df, label_encoding_list)

    # for 1-d
    preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']
    preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']

    df = normalize(df, preference_scores_of_participant, preference_scores_of_partner)
    df.to_csv(output_filename, index = False)

if __name__ == "__main__":
    preprocessing(sys.argv[1], sys.argv[2])