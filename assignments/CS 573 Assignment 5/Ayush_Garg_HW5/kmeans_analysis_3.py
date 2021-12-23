import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import kmeans as KMeans
import functools
import multiprocessing as mp
from itertools import repeat
import kmeans_analysis_1 as KAnalysis
import warnings
warnings.filterwarnings("ignore")


def variation_with_seed(dataset1, dataset2, dataset3, k_list):
    seeds = list(np.random.randint(0, 1000, size=10))
    wc_1 = []
    sc_1 = []
    wc_std_1 = []
    sc_std_1 = []
    wc_2 = []
    sc_2 = []
    wc_std_2 = []
    sc_std_2 = []
    wc_3 = []
    sc_3 = []
    wc_std_3 = []
    sc_std_3 = []
    for k_value in k_list:
        print("Value of K: ", k_value)
        cur_wc_1 = []
        cur_sc_1 = []
        cur_wc_2 = []
        cur_sc_2 = []
        cur_wc_3 = []
        cur_sc_3 = []
        for cur_seed in seeds:
            np.random.seed(cur_seed)
            print("Value of seed: ", cur_seed)
            wc1,sil1=KAnalysis.calculate_wcssd_sc(k_value,dataset1)
            cur_wc_1.append(np.mean(wc1))
            cur_sc_1.append(np.mean(sil1))
            wc2, sil2 = KAnalysis.calculate_wcssd_sc(k_value, dataset2)
            cur_wc_2.append(np.mean(wc2))
            cur_sc_2.append(np.mean(sil2))
            wc3, sil3 = KAnalysis.calculate_wcssd_sc(k_value, dataset3)
            cur_wc_3.append(np.mean(wc3))
            cur_sc_3.append(np.mean(sil3))
        wc_1.append(np.mean(cur_wc_1))
        sc_1.append(np.mean(cur_sc_1))
        wc_std_1.append(np.std(cur_wc_1))
        sc_std_1.append(np.std(cur_sc_1))

        wc_2.append(np.mean(cur_wc_2))
        sc_2.append(np.mean(cur_sc_2))
        wc_std_2.append(np.std(cur_wc_2))
        sc_std_2.append(np.std(cur_sc_2))

        wc_3.append(np.mean(cur_wc_3))
        sc_3.append(np.mean(cur_sc_3))
        wc_std_3.append(np.std(cur_wc_3))
        sc_std_3.append(np.std(cur_sc_3))

    plot_WC_Error_graph(wc_1, wc_std_1)
    plot_SC_Error_graph(sc_1, sc_std_1)
    plot_WC_Error_graph(wc_2, wc_std_2)
    plot_SC_Error_graph(sc_2, sc_std_2)
    plot_WC_Error_graph(wc_3, wc_std_3)
    plot_SC_Error_graph(sc_3, sc_std_3)

def plot_SC_Error_graph(line, error):
    plt.errorbar(k_list, line, yerr=error, marker='^', color='green',ecolor='purple',elinewidth=1, capsize=2, barsabove = True)
    plt.xlabel('K')
    plt.ylabel('SC')
    plt.title('k-means sensitivity')
    plt.show()

def plot_WC_Error_graph(line, error):
    plt.errorbar(k_list, line, yerr=error, marker='^', color='blue',ecolor='red',elinewidth=1, capsize=2, barsabove = True)
    plt.xlabel('K')
    plt.ylabel('WC-SSD')
    plt.title('k-means sensitivity')
    plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    label_list_2=[2,4,6,7]
    label_list_3=[6,7]
    dataset1 = pd.read_csv('digits-embedding.csv', header = None)
    dataset2=KAnalysis.generate_data_subset(dataset1, label_list_2)
    dataset3=KAnalysis.generate_data_subset(dataset1, label_list_3)
    
    k_list=[2,4,8,16,32]
    variation_with_seed(dataset1, dataset2, dataset3, k_list)