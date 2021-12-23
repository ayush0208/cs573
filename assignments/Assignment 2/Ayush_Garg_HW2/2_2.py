import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def partner_success_rate(df, rating_of_partner_from_participant):
    for col in rating_of_partner_from_participant:
        # print("col", col)
        count = len(np.unique(df[[col]]))
        # print(count)
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

def visualize_rating_of_partner(input_filename):
    df = pd.read_csv(input_filename)
    rating_of_partner_from_participant = ['attractive_partner', 'sincere_partner', 'intelligence_parter','funny_partner', 'ambition_partner', 'shared_interests_partner']
    partner_success_rate(df, rating_of_partner_from_participant)

if __name__ == "__main__":
    visualize_rating_of_partner(sys.argv[1])
