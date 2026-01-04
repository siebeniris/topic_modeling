import os
import zarr

import numpy as np

import zarr
from zadu.measures import *
import umap
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


def save_embeddings_to_zarr(embeddings,
                            filepath,
                            chunk_size=(1000, 768),
                            append=True):
    """
    Saves embeddings to a Zarr array on the fly, handling varying first dimensions.

    Args:
        embeddings (np.ndarray): The NumPy array of embeddings to save.
        filename (str): The path to the Zarr array (directory).
        chunk_size (tuple): The chunk size for the Zarr array.  (1000, 768) is a good starting point.
        append (bool): Whether to append to an existing Zarr array or create a new one.
    """

    if os.path.exists(filepath):
        if append:
            try:
                # Open the existing Zarr array in append mode
                zarray = zarr.open(filepath, mode='a')

                # Check if the new embeddings have a compatible shape
                if embeddings.shape[1] != zarray.shape[1]:
                    raise ValueError(
                        f"New embeddings have incompatible shape {embeddings.shape}. "
                        f"Expected number of columns: {zarray.shape[1]}"
                    )

                # Append the new embeddings to the existing array
                zarray.append(embeddings)
                # print(f"Appended {embeddings.shape[0]} embeddings to {filename}")

            except Exception as e:
                print(f"Error appending to Zarr array: {e}")
                print("Consider creating a new Zarr array or checking the data shape.")

        else:
            print(
                f"Zarr array already exists at {filepath}.  Set append=True to append, or delete the existing array.")

    else:
        try:
            # Create a new Zarr array with a resizable first dimension
            zarray = zarr.open(
                filepath,
                shape=(0, embeddings.shape[1]),  # Initial shape with 0 rows
                chunks=chunk_size,
                dtype=embeddings.dtype,
                mode='w'
            )
            zarray.append(embeddings)  # Initialize with the first batch
            # print(f"Created new Zarr array at {filename} with shape {embeddings.shape}")

        except Exception as e:
            print(f"Error creating Zarr array: {e}")


def loading_embeddings(zarr_filepath="data/summaries/overall_summaries_embeds.zarr"):
    print(f"Loading the Embeddings from {zarr_filepath}")

    embeddings = zarr.open(zarr_filepath, mode="r")[:]

    print(f"Loaded embeddings are of shape {embeddings.shape}")
    return embeddings


def eval_embeddings_distortion(reference_embeddings, embeddings,k=50 ):
    # Measure the distortion
    print(f"measuring the distortion between embeddings before and after dimension reduction ...")
    mrre = mean_relative_rank_error.measure(reference_embeddings, embeddings, k=k)
    nd = neighbor_dissimilarity.measure(reference_embeddings, embeddings, k=k)
    tnc = trustworthiness_continuity.measure(reference_embeddings, embeddings, k=k)
    print(f"MRRE {mrre}, Neighbour Dissimilarity{nd}, Trustworthiness Continuity {tnc}")
    

def preprocessing_embeddings(embeddings, metric="euclidean"):
    #### feature scaling, dimension reduction.
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # implement UMAP on the embeddings to reduce the dimensionality to 50. 
    umap_reducer = umap.UMAP(n_neighbors=50, random_state=42, n_components=50, 
                            metric=metric)
    print(f"Reducing embedding dimensionality using umap")
    umap_reduced_embeddings  = umap_reducer.fit_transform(scaled_embeddings)

    # eval_embeddings_distortion(scaled_embeddings, umap_reduced_embeddings)
    # {'mrre_false': np.float64(0.9565518428284852), 
    # 'mrre_missing': np.float64(0.9490148803467386)}
    #  {'neighbor_dissimilarity': np.float64(380.44931331256225)}
    #  {'trustworthiness': np.float64(0.9307933095904293), 
    # 'continuity': np.float64(0.920855377329346)}
    
    return umap_reduced_embeddings, umap_reducer


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



def caclualte_medoid_centroid_for_clusters(cluster_dict, method="medoid"):
    if method == "centroid":
        print("Calculating centroids")
        centroids={}
        for cluster, embedding in cluster_dict.items():
            embedding_array = np.array(embedding)
            cluster_centroid = np.mean(embedding_array, axis=0)
            centroids[cluster]= cluster_centroid
        return centroids

    elif method == "medoid":
        print("Calculating medoids")
        medoids={}
        for cluster, embedding in cluster_dict.items():
            embedding_array = np.array(embedding)
            distances = cdist(embedding_array, embedding_array, metric="euclidean")
            medoid_index = np.argmin(distances.sum(axis=1))
            cluster_medoid = embedding_array[medoid_index]
            medoids[cluster] = cluster_medoid
        return medoids
    
    else:
        print("choosing method in either medoid or centroid")
        return 

        # distance_medoid_centroid = euclidean(cluster_centroid, cluster_medoid)
        # print(f"cluster {cluster} with {embedding_array.shape[0]} members, distance between medoid and centroid {distance_medoid_centroid}")
        #print(f"mean distances in this cluster max: {np.max(distances.mean(axis=1))}, min: {np.min(distances.mean(axis=1))}")
    

