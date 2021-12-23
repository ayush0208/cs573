import pandas as pd
import numpy as np


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

def to_lower(df):
    val = df['field'].values
    count=0
    for value in val:
        if not(value.islower()):
            count+=1 
    df['field'] = df['field'].str.lower()
    return df, count

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

def one_hot_encoding(df, one_hot_encoding_list):
    for i in range (0,df.shape[0]): 
	    df.gender.values[i] = df.gender.values[i].strip()
    unq_gender_val =  df.gender.unique()
    unq_gender_val_sorted = sorted(unq_gender_val)

    for i in range (0,df.shape[0]): 
        df.race.values[i] = df.race.values[i].strip()
    unq_race_val =  df.race.unique()
    unq_race_val_sorted = sorted(unq_race_val)

    for i in range (0,df.shape[0]): 
        df.race_o.values[i] = df.race_o.values[i].strip()
    unq_race_o_val =  df.race_o.unique()
    unq_race_o_val_sorted = sorted(unq_race_o_val)

    for i in range (0,df.shape[0]):
        df.field.values[i] = df.field.values[i].strip()
    unq_field_val =  df.field.unique()
    unq_field_val_sorted = sorted(unq_field_val)
    one_hot_encoding_list = one_hot_encoding_list
    S = pd.Series( {'gender': unq_gender_val_sorted })
    one_hot = pd.get_dummies(S['gender'])

    female_arr = []
    for i in range (0, one_hot.shape[0] - 1):
        female_arr.append(one_hot.female.values[i])
    print ("Mapped vector for female in column gender: ", female_arr)

    S = pd.Series( {'race': unq_race_val_sorted})
    one_hot = pd.get_dummies(S['race'])
    blck_aa_arr = []
    for i in range (0, one_hot.shape[0] - 1):
        blck_aa_arr.append(one_hot['Black/African American'].values[i])
    print ("Mapped vector for Black/African American in column race: ", blck_aa_arr)


    S = pd.Series( {'race_o': unq_race_o_val_sorted})
    one_hot = pd.get_dummies(S['race_o'])
    other_arr = []
    for i in range (0, one_hot.shape[0]-1):
        other_arr.append(one_hot['Other'].values[i])
    print ("Mapped vector for Other in column race_o: ", other_arr)


    S = pd.Series( {'field': unq_field_val_sorted})
    one_hot = pd.get_dummies(S['field'])
    economics_arr = []
    for i in range (0, one_hot.shape[0]-1):
        economics_arr.append(one_hot['economics'].values[i])
    print ("Mapped vector for economics in column field: ", economics_arr)

    df = pd.get_dummies(data=df, columns=['gender', 'race','race_o','field'])
    df.drop(columns = ['gender_male'], axis=1, inplace=True)
    df.drop(columns = ['race_Other'], axis=1, inplace=True)
    df.drop(columns = ['race_o_Other'], axis=1, inplace=True)
    df.drop(columns = ['field_writing: literary nonfiction'], axis=1, inplace=True)
    
    return df

def train_test_split(df):
    test_df = df.sample(random_state=25, frac=0.2)
    train_df = df.drop(test_df.index)
    return train_df, test_df

def split_data(df, train_filename, test_filename):
    train_df, test_df = train_test_split(df)
    train_df.to_csv(train_filename, index = False)
    test_df.to_csv(test_filename, index = False)

def preprocessing():
    num_of_rows = 6500
    df = pd.read_csv('./dating-full.csv', nrows=num_of_rows)
    # for 1-a
    strip_quotes_list = ['race','race_o','field']
    count = strip_quotes(df, strip_quotes_list)

    # for 1-b
    df, count = to_lower(df)
    one_hot_encoding_list = ['gender','race','race_o','field']
    df = one_hot_encoding(df, one_hot_encoding_list)
    
    # for 1-d
    preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']
    preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']

    df = normalize(df, preference_scores_of_participant, preference_scores_of_partner)
    
    split_data(df, 'trainingSet.csv', 'testSet.csv')

if __name__ == "__main__":
    preprocessing()