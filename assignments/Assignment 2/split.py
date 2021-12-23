import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def train_test_split(df):
    test_df = df.sample(random_state=47, frac=0.2)
    train_df = df.drop(test_df.index)
    return train_df, test_df

def split_data(input_filename, train_filename, test_filename):
    df = pd.read_csv(input_filename)
    train_df, test_df = train_test_split(df)
    train_df.to_csv(train_filename, index = False)
    test_df.to_csv(test_filename, index = False)

if __name__ == "__main__":
    split_data(sys.argv[1], sys.argv[2], sys.argv[3])