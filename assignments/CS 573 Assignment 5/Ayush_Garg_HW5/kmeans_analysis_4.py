import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import kmeans as KMeans
import kmeans_analysis_1 as KAnalysis

def plot_cluster_graph(reader,choose_point,belong_dict,dataset_type):
    new_center_dict=defaultdict(list)
    plt.cla()
    plt.clf()
    for id in choose_point:
        belong_cid=belong_dict[id]
        new_center_dict[belong_cid].append(id)
    for key in new_center_dict:
        choose_id=new_center_dict[key]
        result=np.array(reader.iloc[choose_id,2:])
        plt.scatter(result[:, 0], result[:, 1], label=str(key))
    plt.legend()
    plt.savefig('Cluster_'+str(dataset_type)+'.png')

def Calculate_Result(reader,dataset_type, k_value):
    choose_point=np.random.randint(0, len(reader), size=1000)
    start_point = np.random.randint(0, len(reader), size=k_value)
    start_point = np.array(reader.iloc[start_point, 2:])
    center_point = start_point.copy()
    center_point, center_dict, belong_dict = KMeans.kmeans(reader, center_point, 50, k_value)
    NMI = KMeans.Calculate_NMI(reader, center_point, center_dict, belong_dict)
    plot_cluster_graph(reader,choose_point,belong_dict,dataset_type)
    return NMI

if __name__ == '__main__':
    np.random.seed(0)
    label_list_2=[2,4,6,7]
    label_list_3=[6,7]
    dataset1 = pd.read_csv('digits-embedding.csv', header = None)
    dataset2=KAnalysis.generate_data_subset(dataset1, label_list_2)
    dataset3=KAnalysis.generate_data_subset(dataset1, label_list_3)
    k1=8
    k2=4
    k3=2
    NMI1 = Calculate_Result(dataset1,1, k1)
    NMI2 = Calculate_Result(dataset2,2, k2)
    NMI3 = Calculate_Result(dataset3,3, k3)
    print('Dataset1 NMI: %.3f' % NMI1)
    print('Dataset2 NMI: %.3f' % NMI2)
    print('Dataset3 NMI: %.3f' % NMI3)


