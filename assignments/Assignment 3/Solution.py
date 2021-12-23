import pandas as pd 
import numpy as np 

# for 1-b
one_hot_encoding_list = ['gender','race','race_o','field']
one_hot_encoded_data = pd.get_dummies(df, columns = one_hot_encoding_list)

