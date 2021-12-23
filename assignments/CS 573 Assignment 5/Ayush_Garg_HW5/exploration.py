import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import sys


def visualise_digit_raw(filename):
    df = pd.read_csv(filename, header = None)
    for k in range (0,10):
        digit=df[df.iloc[:,1]==k]
        random_id=random.randint(0,len(digit)-1)
        image=np.array(digit.iloc[random_id,2:])
        sub_plot = plt.subplot2grid((2, 5), (int(k/5),k%5))
        image=image.reshape([28, 28])
        sub_plot.imshow(image, cmap='gray_r')
        plt.title('Label: {}'.format(k))
    plt.show()

def visualise_digit_embedding(filename):
    df2 = pd.read_csv(filename, header = None)
    N = df2.shape[0]
    examples =  np.random.randint(0, N, size=1000)
    x_axis = []
    y_axis = []
    class_label = []
    for i in examples:
        class_label.append(df2.loc[i,1])
        x_axis.append(df2.values[i,2])
        y_axis.append(df2.values[i,3])
    data = pd.DataFrame()
    data['x_label'] = x_axis
    data['y_label'] = y_axis
    data['class'] = class_label
    label = np.unique(class_label)
    colors = ['red','green','blue','purple','gray','brown','cyan','orange','magenta', 'yellow']
    for k in range(0,len(label)):
        plt.scatter(data['x_label'][(data['class'] == k)],
                    data['y_label'][(data['class'] == k)],
                marker='o',
                color=colors[k],
                label=k)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    visualise_digit_raw('digits-raw.csv')
    visualise_digit_embedding('digits-embedding.csv')