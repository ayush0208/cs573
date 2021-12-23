import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def data_trend_by_gender(df, preference_scores_of_participant):
    df_male = df[df['gender']==1]
    df_female = df[df['gender']==0]

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

def visualize_preference_score(input_filename):
    df = pd.read_csv(input_filename)
    preference_scores_of_participant = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']
    data_trend_by_gender(df, preference_scores_of_participant)

if __name__ == "__main__":
    visualize_preference_score(sys.argv[1])
