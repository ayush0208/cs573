import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import kmeans as KMeans
import functools
import multiprocessing as mp
from itertools import repeat

def generate_data_subset(reader, label_list):
    new_reader=reader[reader.iloc[:, 1].isin(label_list)]
    return new_reader

def calculate_wcssd_sc(k_value,reader):
    start_point = np.random.randint(0, len(reader), size=k_value)
    start_point = np.array(reader.iloc[start_point, 2:])
    centroids, center_dict, belong_dict = KMeans.kmeans(reader, start_point, 50, k_value)
    WC_SSD = KMeans.Calculate_WCSSD(reader, centroids, belong_dict, k_value, center_dict)
    pool = mp.Pool(6)
    variable = [i for i in range(len(reader))]
    args = list(zip(repeat(reader), repeat(belong_dict), repeat(center_dict), repeat(centroids), variable))
    silhouette_records=pool.starmap(KMeans.silhouette_calculation, args)
    pool.close()
    return WC_SSD,np.mean(silhouette_records)

def plot_graph(k_list,result):
    result = np.array(result)
    _, axs = plt.subplots(3, 2)
    for i in range(3):
        axs[i, 0].plot(k_list, result[:, 2*i], 'C1', marker='o')
        axs[i, 0].set_xlabel('K')
        axs[i, 0].set_ylabel('Within-Cluster SSD')
        axs[i, 0].set_title('Dataset {}'.format(i+1))
        axs[i, 1].plot(k_list, result[:, 2*i+1], 'C2', marker='o')
        axs[i, 1].set_xlabel('K')
        axs[i, 1].set_ylabel('Silhouette Coefficient')
        axs[i, 1].set_title('Dataset {}'.format(i+1))
    plt.show()

def variation_with_K(dataset1, dataset2, dataset3, k_list, result):
    for k_value in k_list:
        print("Value of K: ", k_value)
        wc1,sil1=calculate_wcssd_sc(k_value,dataset1)
        wc2, sil2 = calculate_wcssd_sc(k_value, dataset2)
        wc3, sil3 = calculate_wcssd_sc(k_value, dataset3)
        result.append([wc1,sil1,wc2,sil2,wc3,sil3])
    return result

if __name__ == '__main__':
    np.random.seed(0)
    label_list_2=[2,4,6,7]
    label_list_3=[6,7]
    dataset1 = pd.read_csv('digits-embedding.csv', header = None)
    dataset2=generate_data_subset(dataset1, label_list_2)
    dataset3=generate_data_subset(dataset1, label_list_3)
    k_list=[2,4,8,16,32]
    result=[]
    result = variation_with_K(dataset1, dataset2, dataset3, k_list, result)
    plot_graph(k_list,result)