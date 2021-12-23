import pandas as pd
import numpy as np
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import kmeans as KMeans
import multiprocessing as mp
import scipy.cluster.hierarchy as sch
from itertools import repeat

def plot_graph_k_variation(k_list, wc_list,sc_list, method,given_label):
    plt.cla()
    plt.clf()
    fig_path=method+'_WC.jpg'
    plt.plot(k_list, wc_list,label=given_label)
    plt.xlabel('K value')
    plt.ylabel('WC SSD')
    plt.legend()
    plt.savefig(fig_path)
    plt.cla()
    plt.clf()
    fig_path=method+'_SC.jpg'
    plt.plot(k_list, sc_list,label=given_label)
    plt.xlabel('K value')
    plt.ylabel('Silhouette Coefficient')
    plt.legend()
    plt.savefig(fig_path)

def get_center_points(all_data,cluster):
    center_dict=defaultdict(list)
    belong_dict={}
    for i in range(len(all_data)):
        belong_dict[i]=cluster[i]-1
        center_dict[belong_dict[i]].append(i)
    keys=list(center_dict.keys())
    keys.sort()
    center_point=[]
    for key in keys:
        use_id=center_dict[key]
        tmp_mean_coord=np.array(np.mean(all_data.iloc[use_id,2:]))
        center_point.append(tmp_mean_coord)
    return center_point,center_dict,belong_dict

def get_dendrograms(all_data,linkage_type, k_list):
    plt.cla()
    plt.clf()
    points = np.array(all_data.iloc[:, 2:])
    disMat = sch.distance.pdist(points, 'euclidean')
    Z = sch.linkage(disMat, method=linkage_type)
    P = sch.dendrogram(Z)
    plt.savefig('dendrogram_'+str(linkage_type)+'.png')
    wc_list=[]
    sc_list=[]
    for k in k_list:
        cluster=sch.fcluster(Z, t=k, criterion='maxclust')
        centroids,center_dict,belong_dict=get_center_points(all_data,cluster)
        WC_SSD = KMeans.Calculate_WCSSD(all_data, centroids, belong_dict, k, center_dict)
        wc_list.append(WC_SSD)
        pool = mp.Pool(6)
        variable = [i for i in range(len(all_data))]
        args = list(zip(repeat(all_data), repeat(belong_dict), repeat(center_dict), repeat(centroids), variable))
        silhouette_records=pool.starmap(KMeans.silhouette_calculation, args)
        pool.close()
        sc_list.append(np.mean(silhouette_records))
    plot_graph_k_variation(k_list, wc_list,sc_list, linkage_type,linkage_type)
    return wc_list,sc_list

def calculate_NMI(all_data,linkage_type, k_value):
    plt.cla()
    plt.clf()
    points = np.array(all_data.iloc[:, 2:])
    disMat = sch.distance.pdist(points, 'euclidean')
    Z = sch.linkage(disMat, method=linkage_type)
    P = sch.dendrogram(Z)
    cluster=sch.fcluster(Z, t=k_value, criterion='maxclust')
    centroids,center_dict,belong_dict=get_center_points(all_data,cluster)
    NMI = KMeans.Calculate_NMI(all_data, centroids, center_dict, belong_dict)
    return NMI

def plot_dendrograms(reader):
    list_result=[]
    k_list=[2,4,8,16,32]
    clusters = 10
    for k in range(clusters):
        new_Reader=reader[reader.iloc[:,1]==k]
        choose_point = np.random.randint(0, len(new_Reader), size=10)
        result=new_Reader.iloc[choose_point,:]
        list_result.append(result)
    all_data = pd.concat(list_result)
    wc_list1, sc_list1=get_dendrograms(all_data,'single', k_list)
    wc_list2, sc_list2=get_dendrograms(all_data, 'complete', k_list)
    wc_list3, sc_list3=get_dendrograms(all_data, 'average', k_list)

def get_NMI_value(reader, k_value):
    np.random.seed(2)
    list_result=[]
    clusters = 10
    for k in range(clusters):
        new_Reader=reader[reader.iloc[:,1]==k]
        choose_point = np.random.randint(0, len(new_Reader), size=10)
        result=new_Reader.iloc[choose_point,:]
        list_result.append(result)
    all_data = pd.concat(list_result)
    NMI_single=calculate_NMI(all_data,'single', k_value)
    NMI_complete=calculate_NMI(all_data, 'complete', k_value)
    NMI_average=calculate_NMI(all_data, 'average', k_value)
    print("NMI_single:", NMI_single)
    print("NMI_complete:", NMI_complete)
    print("NMI_average:", NMI_average)

if __name__ == "__main__":
    reader = pd.read_csv('digits-embedding.csv', header = None)
    plot_dendrograms(reader)
    # reader = pd.read_csv('digits-embedding.csv', header = None)
    get_NMI_value(reader, 16)