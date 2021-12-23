import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# for 1-a problem
def contains_single_quote(s):
    if s.startswith("'") or s.endswith("'"):
        return True
    else:
        return False

def strip_quotes(df):
    strip_quotes_list = ['race','race_o','field']
    count = 0
    for index in df.index:
        for col in strip_quotes_list:
            if contains_single_quote(df.loc[index, col]):
                count+=1
                df.loc[index, col] = df.loc[index, col].replace('\'','')
    print("Quotes removed from " + str(count) + " cells")

# for 1-b problem
def to_lower(df):
    val = df['field'].values
    count=0
    for value in val:
        if not(value.islower()):
            count+=1 
    df['field'] = df['field'].str.lower()
    print("Standardized " + str(count) + " cells to lower case")
    return df

# for 1-c problem
def label_encoding(df):
    label_encoding_list = ['gender','race','race_o','field']
    encoding_dict = {}
    for col in label_encoding_list:
        unique_val = np.unique(df[col].values)
        val_dict = {}
        encoding_value = 0
        for val in unique_val:
            val_dict[val]=encoding_value
            encoding_value+=1
            df[col].replace(val, encoding_value, inplace=True)
        encoding_dict[col] = val_dict
    print("Value assigned for male in column gender:", encoding_dict['gender']['male'])
    print("Value assigned for European/Caucasian-American in column race:", encoding_dict['race']['European/Caucasian-American'])
    print("Value assigned for Latino/Hispanic American in column race o:", encoding_dict['race_o']['Latino/Hispanic American'])
    print("Value assigned for law in column field:", encoding_dict['field']['law'])

# for 1-d problem
preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']
preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']
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
def normalize(df):
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
        print("Mean of "+ col + ": " ,round(df[[col]].mean()[0], 2) )
    for col in preference_scores_of_partner:
        print("Mean of "+ col + ": " , round(df[[col]].mean()[0], 2) )    

# for 2-a problem
def data_trend_by_gender(df):
    df_male = df[df['gender']=='male']
    df_female = df[df['gender']=='female']

    male = []
    for col in preference_scores_of_participant:
        mean = df_male[[col]].mean()[0]
        # print("Mean of "+ col + ": " , mean)
        male.append(mean)

    female = []
    for col in preference_scores_of_participant:
        mean = df_female[[col]].mean()[0]
        # print("Mean of "+ col + ": " , mean)
        female.append(mean)
        
    barWidth = 0.25
    plt.subplots(figsize =(15, 8))
    
    br1 = np.arange(len(male))
    br2 = [x + barWidth for x in br1]
    
    # Make the plot
    plt.bar(br1, male, color ='r', width = barWidth,
            edgecolor ='grey', label ='Male')
    plt.bar(br2, female, color ='b', width = barWidth,
            edgecolor ='grey', label ='female')
    
    # Adding Xticks
    plt.xlabel('Attributes', fontweight ='bold', fontsize = 15)
    plt.ylabel('Mean', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(male))],
            preference_scores_of_participant)
    plt.legend()
    plt.show()

# for 2-b problem
def partner_success_rate(df):
    for col in rating_of_partner_from_participant:
        print("col", col)
        count = len(np.unique(df[[col]]))
        print(count)
        col_values = np.unique(df[[col]])
        success = []
        for val in col_values:
            den = df[df[col]==val].count()[0]
            num = df[df[col]==val]['decision'].sum()
            success_rate = num/den
            success.append(success_rate)
        plt.scatter(col_values,success)
        plt.title(col)
        plt.xlabel("Values in " + col)
        plt.ylabel("Success Rate")
        plt.show()

# for 3 problem
def binning_continuous_valued_columns(df):
    for col in continuous_valued_columns:
        # print(col)
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
        # print(min_val)
        # print(max_val)
        bins = np.linspace(min_val,max_val,6)
        df.loc[df[col] < min_val, col] = max_val
        df.loc[df[col] > max_val, col] = max_val
        df[col] = pd.cut(df[col],bins, include_lowest=True)
        print(col, df[col].value_counts().sort_index().values)
        # s = df[col].value_counts()
        # print(s)

# for 4 problem
def train_test_split(df):
    test_df = df.sample(random_state=47, frac=0.2)
    train_df = df.drop(test_df.index)
    return train_df, test_df

# for 5 problem
def prior_class_probability(df):
    positive = len(df[df['decision']==1])
    negative = len(df) - positive
    prob_positive = positive/len(df)
    prob_negative = negative/len(df)
    return prob_positive, prob_negative

def create_lookup_table(df, target_col):
    lookup_table = {}
    value_counts = df[target_col].value_counts().sort_index()
    lookup_table['class_name'] = value_counts.index.to_numpy()
    lookup_table['class_count'] = value_counts.values

    data_columns = df.drop(target_col, axis=1).columns
    for col in data_columns:
        lookup_table[col] = {}

        counts = df.groupby(target_col)[col].value_counts()
        df_counts = counts.unstack(target_col)
        if df_counts.isna().any(axis=None):
            df_counts.fillna(value=0, inplace = True)
            df_counts+=1  # laplace smoothing I guess
        df_probabilities = df_counts/df_counts.sum()
        for val in df_probabilities.index:
            probability = df_probabilities.loc[val].to_numpy()
            lookup_table[col][val] = probability
    return lookup_table

def predict(row, result):
    class_estimates = result['class_count']
    row = row[:-1] # can be removed after we pass the dataset without target label column
    for feature in row.index:
        try:
            value = row[feature]
            prob = result[feature][value]
            class_estimates = class_estimates * prob
        except KeyError:
            continue

    class_estimates
    max_class_index = class_estimates.argmax()
    prediction = result['class_name'][max_class_index]
    return prediction

def accuracy(result, df, target_col):
    predictions = df.apply(predict, axis=1, args=(result,))
    correct_predictions = predictions == df[target_col]
    return correct_predictions.mean()

# def nbc(t_frac):
#     df = pd.read_csv('./trainingSet.csv')
#     train_df = df.sample(random_state=47, frac=t_frac)
#     test_df = pd.read_csv('./testSet.csv')
#     prob_table = create_lookup_table(train_df, 'decision')
#     print("Training Accuracy: ",round(accuracy(prob_table, train_df), 2))
#     print("Testing Accuracy: ", round(accuracy(prob_table, test_df), 2))

def nbc(t_frac):
    df = pd.read_csv('./trainingSet.csv')
    train_df = df.sample(random_state=47, frac=t_frac)
    target_col = 'decision'
    test_df = pd.read_csv('./testSet.csv')
    prob_table = create_lookup_table(train_df, target_col)
    return prob_table, train_df, test_df

# for 5-b problem
def effect_of_bins(input_filename, bin_value_list):
    # bin_value_list = [2,5,10,50,100,200]
    for bin in bin_value_list:
        print("Bin value", bin)
        df = pd.read_csv(input_filename)
        binning_continuous_valued_columns(df, continuous_valued_columns, preference_scores_of_participant, preference_scores_of_partner, bin)
        train_df, test_df = train_test_split(df)
        train_df.to_csv('./trainingSet.csv', index = False)
        test_df.to_csv('./testSet.csv', index = False)
        # print(len(train_df), len(test_df))
        resultant_table, train_df, test_df = nbc(1)
        # print("Hello1")
        print("Training Accuracy: ",round(accuracy(resultant_table, train_df), 2))
        print("Testing Accuracy: ", accuracy(resultant_table, test_df))
        # print("Hello2")
        # print("Testing Accuracy: ", round(accuracy(resultant_table, test_df), 2))

bin_value_list = [2,5,10,50,100,200]
input_filename = './dating.csv'

# for 5-c problem
def effect_of_fraction(input_filename, fraction_list):
    for frac in fraction_list:
    # f = 0.2
        print("Frac", frac)
        df = pd.read_csv(input_filename)
        binning_continuous_valued_columns(df, continuous_valued_columns, preference_scores_of_participant, preference_scores_of_partner, 5)
        train_df, test_df = train_test_split(df)
        train_df.to_csv('./trainingSet.csv', index = False)
        test_df.to_csv('./testSet.csv', index = False)
        print("Now actual data samples for train and test")
        resultant_table, train_df, test_df = nbc(frac)
        print(len(train_df), len(test_df))
        print("Training Accuracy: ",round(accuracy(resultant_table, train_df), 2))
        print("Testing Accuracy: ", accuracy(resultant_table, test_df))
    