import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import functools
import multiprocessing as mp
from itertools import repeat

def calc_euclidean_distance(A, B):
    A_square = np.reshape(np.sum(A * A, axis=1), (A.shape[0], 1))
    B_square = np.reshape(np.sum(B * B, axis=1), (1, B.shape[0]))
    AB = A @ B.T
    C = A_square + B_square - 2 * AB
    C[C<0] = 0
    return np.sqrt(C)

def Calculate_WCSSD(reader,centroids,belong_dict, clusters, center_dict):
        distance = 0
        for i in range(clusters):
            group_member_ids = center_dict[i]
            A = reader.iloc[group_member_ids,2:].values
            B = centroids[i].reshape(1, -1)
            distance += np.sum(calc_euclidean_distance(A, B)**2)
        return distance


def calculate_error(a,b):
    error = np.sqrt(np.sum((a-b)**2))
    return error 

def dist(a, b):
    return np.linalg.norm(a - b, axis=1)

def Calculate_Belonging(example,centroids):
    feature_example=(example.iloc[2:]).values
    feature_example = feature_example.reshape(-1, len(feature_example))
    distances = calc_euclidean_distance(feature_example, centroids)
    cluster_index = np.argmin(distances)    
    return cluster_index

def get_clusters(X, centroids):
    X = X.iloc[:,2:].values
    distance_matrix = calc_euclidean_distance(X, centroids)
    closest_cluster_ids = np.argmin(distance_matrix, axis=1)
    return closest_cluster_ids

def kmeans(reader,centroids, max_iter, num_of_clusters):
    dataset_length = len(reader)
    new_reader = reader.copy()
    iter=0
    belong_dict={}
    center_dict=defaultdict(list)
    while iter<max_iter:
        clusters = get_clusters(reader, centroids)
        new_reader['clusterNo'] = clusters

        new_centroids=[]
        for i in range(0,num_of_clusters):
            centroid = new_reader[new_reader['clusterNo'] == i].mean()
            new_centroids.append([centroid[2], centroid[3]])                
        centroids = np.array(new_centroids)
        iter+=1

    for i in range(dataset_length):
        belong_dict[i]= new_reader.iloc[i,4]
        center_dict[belong_dict[i]].append(i)
    return centroids,center_dict,belong_dict

def silhouette_calculation(reader,belong_dict,center_dict, centroids, i):
    instance_group=belong_dict[i]
    group_member_ids=center_dict[instance_group]

    if len(group_member_ids)==1:
        return 0

    instance_features=reader.iloc[i,2:]
    example_features = instance_features.values
    example_features = example_features.reshape(-1, len(instance_features))
    members_A = reader.iloc[group_member_ids,2:].values
    a_value = np.mean(calc_euclidean_distance(members_A, example_features))
    closest_centroid = np.argsort(calc_euclidean_distance(np.array(centroids), example_features), axis = 0)[1][0]
    members_B = reader.iloc[center_dict[closest_centroid], 2:].values
    b_value = np.mean(calc_euclidean_distance(members_B, example_features))
    sc_i = (b_value-a_value)/max(a_value,b_value)
    return sc_i

def calculate_entropy(prob):
    return -prob*np.log2(prob)

def Calculate_NMI(reader,centroids,center_dict,belong_dict):
    possible_label=np.unique(reader[reader.columns[1]])
    prob_y_dict={}
    h_y=0
    for label in possible_label:
        prob=reader[reader.columns[1]].value_counts()[label] /len(reader)
        prob_y_dict[label]=prob
        h_y+=calculate_entropy(prob)
    prob_c_dict = {}
    h_c=0
    for key in center_dict:
        current_cluster=center_dict[key]
        prob=len(current_cluster)/len(reader)
        prob_c_dict[key]=prob
        h_c+=calculate_entropy(prob)
    info_loss_entropy=0
    for key in center_dict:
        current_cluster = center_dict[key]
        tmp_dict = {}
        for example_id in current_cluster:
            example_label=reader.iloc[example_id,1]
            if example_label not in tmp_dict:
                tmp_dict[example_label]=1
            else:
                tmp_dict[example_label]+= 1
        for tmp_key in tmp_dict:
            prob1=prob_y_dict[tmp_key]
            prob2=prob_c_dict[key]
            prob3=tmp_dict[tmp_key]/len(reader)
            info_loss_entropy+=prob3*np.log2(prob3/(prob1*prob2))
    info_gain=info_loss_entropy
    return info_gain/(h_y+h_c)

def call_KMeans():
    np.random.seed(0)
    file_name = sys.argv[1]
    clusters = int(sys.argv[2])
    reader = pd.read_csv(file_name, header = None)
    start_point=np.random.choice(len(reader), clusters, replace =False)
    centroids=np.array(reader.iloc[start_point,2:])
    centroids,center_dict,belong_dict=kmeans(reader,centroids, 50, clusters)
    WC_SSD=Calculate_WCSSD(reader,centroids,belong_dict, clusters, center_dict)
    print('WC-SSD: %.3f' % WC_SSD)
    pool = mp.Pool(6)
    variable = [i for i in range(len(reader))]
    args = list(zip(repeat(reader), repeat(belong_dict), repeat(center_dict), repeat(centroids), variable))
    silhouette_records=pool.starmap(silhouette_calculation, args)
    pool.close()
    print('SC: %.2f'%np.mean(silhouette_records))
    NMI = Calculate_NMI(reader, centroids, center_dict, belong_dict)
    print('NMI: %.3f' % NMI)

if __name__ == "__main__":
    call_KMeans()