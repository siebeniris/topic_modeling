import warnings
import argparse
import os
import numpy as np
import pandas as pd 

from tqdm import tqdm
from zadu.measures import *
import hdbscan
from collections import defaultdict, Counter
from dotenv import load_dotenv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn import metrics

from pipeline.utils.embedding_utils import (loading_embeddings,
                                            eval_cluster)
from pipeline.utils.sampling import stratified_max_min_sampling_proportional_precomputed

import pickle



def eval_cluster(preprocessed_embeddings, final_labels, clustering="hdbscan"):
    # Silhouette, Davies Bouldin, Calinski Harabasz are better at evaluating convex clusters.
    # evaluation of the clusters, [-1,1] , 1 is better.
    if clustering == "hdbscan":
        print(f"Evaluating the hdbscan clusters using using masked embeddings and labels")
        # filter out the noise points frist
        print(f" embeddings {preprocessed_embeddings.shape} and labels {final_labels.shape}")
        core_samples_mask = np.zeros_like(final_labels, dtype=bool)
        core_samples_mask[final_labels > -1] = True
        
        preprocessed_embeddings = preprocessed_embeddings[core_samples_mask]
        final_labels = final_labels[core_samples_mask]
        print(f" embeddings {preprocessed_embeddings.shape} and labels {final_labels.shape}")
    
    silhouette_score = metrics.silhouette_score(preprocessed_embeddings, final_labels, metric="euclidean")
    # values closer to zero indicate a better partition
    db_index = metrics.davies_bouldin_score(preprocessed_embeddings, final_labels)
    # values calinski harabasz 
    # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    ch_index = metrics.calinski_harabasz_score(preprocessed_embeddings, final_labels)
    print(f"Eval clustering: Silhouette {silhouette_score} | Davies-Bouldin {db_index} |  Calinski-Harabasz {ch_index} .")
    
    return silhouette_score, db_index, ch_index

# load the classifier, scaler, umap reducer.
    
with open("output/knn_cosine_n10_distance_subcluster.pkl", "rb") as f:
    knn_sub_cluster = pickle.load(f)

with open("output/scaler_task_embeddings.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("output/umap_reducer_task_embeddings.pkl", "rb") as f:
    umap_reducer = pickle.load(f)   
    
    
# some new data from May
new_data = loading_embeddings("data/database/new_data.zarr")
scaled_data = scaler.transform(new_data)
processed_data = umap_reducer.transform(scaled_data)


X_subcluster_train_filename = 'data/database/knn_data/sub_cluster/X_train.npy'
X_subcluster_test_filename = 'data/database/knn_data/sub_cluster/X_test.npy'
y_subcluster_train_filename = 'data/database/knn_data/sub_cluster/y_train.npy'
y_subcluster_test_filename = 'data/database/knn_data/sub_cluster/y_test.npy'


X_subcluster_train= np.load(X_subcluster_train_filename)
X_subcluster_test= np.load(X_subcluster_test_filename)
y_subcluster_train= np.load(y_subcluster_train_filename)
y_subcluster_test= np.load(y_subcluster_test_filename)


print(knn_sub_cluster.score(X_subcluster_test, y_subcluster_test))


# baseline for comparison.
y_subcluster_pred = knn_sub_cluster.predict(X_subcluster_test)
acc = knn_sub_cluster.score(X_subcluster_test, y_subcluster_test)
f1_weighted = f1_score(y_subcluster_test, y_subcluster_pred, average="weighted")
print(f", eval acc {acc}, weighted f1 {f1_weighted}")


# incorporating new data.
preds = knn_sub_cluster.predict(processed_data)
probs = knn_sub_cluster.predict_proba(processed_data)
probs = np.max(probs, axis=1)

eval_cluster(X_subcluster_train, y_subcluster_train, "other")

# filtere the embeddings with threshold.
def filter_embeds(processed_data, probs, threshold):
    X_new_filtered =processed_data[probs>=threshold]
    y_filtered = preds[probs>=threshold]
    eval_cluster(X_new_filtered, y_filtered, "other")
    print(len(y_filtered))
    return X_new_filtered, y_filtered


# do an iteration
for t in [0.5, 0.6, 0.7, 0.8,0.9,1]:
    print(f"threshold {t}...")
    filter_embeds(processed_data, probs, t)
    
    
# 0.6 is the best 
X_new_filtered, y_new_filtered = filter_embeds(processed_data, probs,0.6)


X_merged = np.concatenate((X_subcluster_train, X_new_filtered), axis=0)
X_merged = np.concatenate((y_subcluster_train, y_new_filtered), axis=0)




# Perform stratified Max-Min sampling with proportional allocation
for p in np.arange(0.1,1,0.1):
    total_sample_size = int(X_merged.shape[0]*p)
    print(f"sampling {total_sample_size}")
    
    X_sampled, y_sampled, sampled_indices = stratified_max_min_sampling_proportional_precomputed(
        X_merged, X_merged, total_sample_size, metric='euclidean', random_state=42
    )
    
    # Print the shape of the sampled embeddings and labels
    print("Sampled Embeddings Shape:", X_sampled.shape)
    print("Sampled Labels Shape:", y_sampled.shape)
    eval_cluster(X_sampled, y_sampled, 'other')
    
    knn = KNeighborsClassifier(metric="cosine", n_neighbors=10, weights="distance")
    knn.fit(X_sampled, y_sampled)
    y_subcluster_pred = knn.predict(X_subcluster_test)
    acc = knn.score(X_subcluster_test, y_subcluster_test)
    f1_weighted = f1_score(y_subcluster_test, y_subcluster_pred, average="weighted")
    print(f"sampling {total_sample_size}, eval acc {acc}, weighted f1 {f1_weighted}")

    
    print("*"*40)
