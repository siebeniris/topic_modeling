import warnings
import argparse
import os
import numpy as np
import pandas as pd 

from tqdm import tqdm
from zadu.measures import *
import hdbscan
from collections import defaultdict, Counter

from pipeline.utils.embedding_utils import (loading_embeddings,
                                            save_embeddings_to_zarr,
                                            preprocessing_embeddings,
                                            eval_cluster,
                                            caclualte_medoid_centroid_for_clusters
                                            )
from pipeline.utils.utils import load_csv_file

warnings.filterwarnings("ignore")

# Clustering using HDBSCAN


def get_hdbscan_cluster_labels(embeddings, min_cluster_size, min_samples=100):
    # min_cluster_size: smallest size grouping that is considered to be a cluster
    # min_samples: the higher, the noiser.
    # cluster_selection_epsilon: t ensures that clusters below the given threshold are not split up any further
    clusterer = hdbscan.HDBSCAN(min_samples=min_samples, 
                             min_cluster_size=min_cluster_size,
                             metric="euclidean"
                             )
    labels = clusterer.fit_predict(embeddings)
    probs = clusterer.fit(embeddings).probabilities_
    return labels, probs


def get_cluster_dict(embeddings, labels):
    cluster_dict = defaultdict(list)
    for embed, label in tqdm(zip(embeddings, labels)):
        cluster_dict[label].append(embed)
    return cluster_dict


def iterative_clustering(reporter,
                         preprocessed_embeddings: np.array,
                         initial_cluster_size: int = 50,
                         min_samples:int =15,
                         refinement: bool = True,
                         ):
    print("iterative clustering...")
    # initialize samples.
    initial_samples = preprocessed_embeddings.shape[0]

    if initial_cluster_size > 1:
        initial_cluster_size = initial_cluster_size
    else:
        initial_cluster_size = int(np.sqrt(initial_samples))
    print(f" initial cluster size {initial_cluster_size}")

    # initialize the cluster size.
    cluster_size = initial_cluster_size

    # update the cluster labels after different steps of clustering
    final_labels = np.full(initial_samples, -1)
    num_clusters_so_far = 0  # keep track of the clusters

    # 1. Initial clustering with hdbscan.
    labels, _ = get_hdbscan_cluster_labels(preprocessed_embeddings, min_cluster_size=cluster_size, min_samples=min_samples)
    final_labels = labels.copy()
    num_clusters_so_far = len(set(final_labels)) - \
        (1 if -1 in final_labels else 0)
    print(f"num clusters so far: ", num_clusters_so_far)

    # assign scores as the initial scores.
    sl_score, db_ind, ch_ind =  eval_cluster(preprocessed_embeddings, final_labels, "hdbscan")
    # reporter.write(f"initial_samples: {initial_samples}, min_cluster_size: {cluster_size}, min_samples: {min_samples}\n")
    reporter.write(f"round 0 | cluster size: {num_clusters_so_far} | sil_score: {sl_score}, db_index: {db_ind}, ch_index: {ch_ind}\n")

    # Find noise points after initial clustering, where the labels equal to -1.
    noise_indices = np.where(final_labels == -1)[0]
    noise_points = preprocessed_embeddings[noise_indices]
    print(f"noise points {len(noise_points)}")

    # Iterative addition of new clusters.
    # Add check for empty noise_points
    round = 0
    
    while refinement and len(noise_points) > 0:
        last_cluster_size = cluster_size
        
        # adjsut the cluster size.
        cluster_size = int(np.sqrt(len(noise_points)))
        
        # Refine clustering on noise points with adopted cluster size.
        labels, _ = get_hdbscan_cluster_labels(noise_points, 
                                            min_cluster_size=cluster_size, 
                                            min_samples=min_samples)
        
        # Update final labels with new cluster assignments
        # Only consider newly clustered points
        new_clusters = labels[labels != -1]
        # Indices of newly clustered points in original data
        new_cluster_indices = noise_indices[labels != -1]

        new_final_labels = final_labels.copy()
        # Assign new labels, offsetting by the number of clusters so far
        for i, index in enumerate(new_cluster_indices):
            new_final_labels[index] = new_clusters[i] + num_clusters_so_far

        # Update the number of clusters
        num_clusters_so_far = len(set(new_final_labels)) - \
                (1 if -1 in new_final_labels else 0)
                
        # evaluate:
        round +=1 
        
        # TODO: Adopt this metric for HDBSCAN, because this is more for convex clusters.
        print(f"round{round}, num clusters so far: {num_clusters_so_far}")
        sil_score_refine, db_index_refine, ch_index_refine = eval_cluster(preprocessed_embeddings, new_final_labels, "hdbscan")
        reporter.write(f"round {round} | cluster size: {num_clusters_so_far} | sil_score: {sl_score}, db_index: {db_ind}, ch_index: {ch_ind}\n")
        
        if sil_score_refine > sl_score and db_index_refine < db_ind and ch_index_refine > ch_ind:
            sl_score = sil_score_refine
            db_ind = db_index_refine
            ch_ind = ch_index_refine
            
            final_labels = new_final_labels
            refinement = True
            # Update noise points for the next iteration
            noise_indices = np.where(final_labels == -1)[0]
            noise_points = preprocessed_embeddings[noise_indices]
            print(f"noise points {len(noise_points)}")
            
            # Check for convergence, the cluster size has not changed
            if cluster_size == last_cluster_size:
                refinement = False
        else:
            # if no improvement
            refinement = False 

        
        
    # noise ratio
    noise_ratio = len(final_labels[final_labels == -1])/len(final_labels)
    print(f"noise ratio : {noise_ratio*100} %")
    print("label distribution : ", np.unique(final_labels, return_counts=True))

    # write down the labels for clustering.
    # if len(final_labels) == len(df_summary):
    #     df_summary["cluster_id"] = final_labels
    #     df_summary.to_csv(outputfile, index=False)

    sil_score, db_index, ch_index = eval_cluster(
        preprocessed_embeddings, final_labels, "hdbscan")

    # create a dictionary with label as keys, embeddings as values.
    cluster_dict = get_cluster_dict(preprocessed_embeddings, final_labels)

    reporter.write(f"after iteration {round}: min_cluster_size: {cluster_size}, min_samples: {min_samples}, noise samples {noise_points}, noise ratio {noise_ratio}\n")
    reporter.write(f"initial clustering: cluster size: {num_clusters_so_far} | sil_score: {sl_score}, db_index: {db_ind}, ch_index: {ch_ind}\n")

    return final_labels, cluster_dict, sil_score, db_index, ch_index


def refining_clustering(hdbscan_labels,
                        cluster_dict,
                        embeddings,
                        reporter, 
                        refining_method: str):
    """Refining clusters based on the initial cluster created from iterative HDBSCAN.
    The clusters are refined using either centroid or medoid method.
    The clusters are refined based on the initial scores of evaluating clusters.

    Args:
        cluster_dict (_type_): cluster dictionary with label as keys and embeddings as values.
        embeddings (_type_): embeddings to be clustered.
        intial_sil_score (_type_): _description_
        initial_db_index (_type_): _description_
        initial_ch_index (_type_): _description_
        reporter (_type_): writer to write down the evaluation scores.
        refining_method (str): _description_

    Returns:
        _type_: _description_
    """
    # refining clusters based on the initial scores of evaluating clusters...
    assert refining_method in ["centroid", "medoid"]
    reporter.write(f"******* refining clusters using {refining_method} method *******\n")

    # intialize for calculating the centroids and medoids for clustering.
    cluster_dict_refine = cluster_dict

    # the scores for evaluating the clustering
    # sil_score = intial_sil_score
    # db_index = initial_db_index
    # ch_index = initial_ch_index
    sil_score = 0
    db_index = np.inf
    ch_index = 0

    # refining argument
    refining = True
    labels_by_refinement = None

    counter = 1
    while refining:
        print("refining clustering...")
        refined_labels_by_refine = []

        cluster_dict_refine = caclualte_medoid_centroid_for_clusters(
            cluster_dict_refine, refining_method)
        
        # put embeddings to the closest cluster.
        for embedding in embeddings:
            
            distances = {cluster: np.linalg.norm(
                embedding - centroid) for cluster, centroid in cluster_dict_refine.items()}
            # any constraint that it also should be 
            refined_labels_by_refine.append(min(distances, key=distances.get))

        # evaluating using silhouette, db and ch indices.
        print(f"evaluating after round {counter}...")

        sil_score_refine, db_index_refine, ch_index_refine = eval_cluster(
            embeddings, np.array(refined_labels_by_refine), "hdbscan")
        reporter.write(f"round {counter} | sil_score: {sil_score_refine}, db_index: {db_index_refine}, ch_index: {ch_index_refine}\n")
        
        # Whether new clusters should be created and formed.
        # Using the cluster metrics to measure.
        if sil_score_refine > sil_score and db_index_refine < db_index and ch_index_refine > ch_index:
            sil_score = sil_score_refine
            db_index = db_index_refine
            ch_index = ch_index_refine
            refining = True

            if refined_labels_by_refine:
                labels_by_refinement = refined_labels_by_refine
                cluster_dict_refine = get_cluster_dict(
                    embeddings, labels_by_refinement)
                counter += 1 
        else:
            refining = False
            if labels_by_refinement:
                print(f"outputing labels by {refining_method} refinement")
                return labels_by_refinement
            else: 
                return hdbscan_labels


def load_and_process_data_for_clustering(data_path, content):
    assert content in ["task", "subject", "summary"]

    zarr_filepath = os.path.join(data_path, f"embeddings_mpnet_{content}.zarr")
    summary_file = os.path.join(data_path, f"embeddings_mpnet_{content}.csv")
    processed_embedding_path = os.path.join(data_path, f"embeddings_mpnet_{content}_processed.zarr")
    
    df_summary = load_csv_file(summary_file, content)
    print(f"Loaded the summary file {summary_file} with length {len(df_summary)}.")
    
    # process once and save to locally
    if os.path.exists(processed_embedding_path):
        print(f"loading the processed embeddings from {processed_embedding_path}")
        preprocessed_embeddings = loading_embeddings(processed_embedding_path)
    else:
        print(f"processing the embeddings from {zarr_filepath}")
        embeddings = loading_embeddings(zarr_filepath)
        preprocessed_embeddings, umap_reducer = preprocessing_embeddings(embeddings)
        print(f"Saving the processed embeddings to {processed_embedding_path}")
        save_embeddings_to_zarr(preprocessed_embeddings, processed_embedding_path)

    return df_summary, preprocessed_embeddings
    
    

def main(data_path="data/summaries/gpt-4.1-nano_20250508_170432",
         content="subject", 
         mode = "exp",
         min_clusters_size=150,
         min_samples_size=5):

    assert content in ["task", "subject", "summary"]

    df_summary, preprocessed_embeddings = load_and_process_data_for_clustering(data_path, content)

    # TODO: Intial step to get the best initial clusters to work with.
    if mode == "exp":
        outputdir = os.path.join(data_path, f"exp_{content}")
        os.makedirs(outputdir, exist_ok=True)
        
        reporter = open(os.path.join(outputdir, f"hdbscan_report_{content}.txt"), "w+")

        distributed_list = []
        initial_cluster_size_list = []
        min_samples_list = []
        sil_score_list = []
        db_index_list = []
        ch_index_list = []
        for initial_cluster_size in [50, 100, 150, 200, 250, 300]:
            for min_samples in [5, 10, 20, 30, 50, 70, 100, 150 ]:
                print(f"****************** initial cluster size {initial_cluster_size} ******************")
                cluster_labels, _, sil_score, db_index, ch_index = \
                            iterative_clustering(reporter,
                            preprocessed_embeddings=preprocessed_embeddings,
                            initial_cluster_size=initial_cluster_size,
                            min_samples=min_samples,
                            refinement=True)
                
                distributed_labels_dict = {int(x):y for x,y in dict(Counter(cluster_labels)).items()}
                print(f"initial cluster size {initial_cluster_size} | min_samples {min_samples} | label distribution: {distributed_labels_dict}")
                print(f"sil_score: {sil_score}, db_index: {db_index}, ch_index: {ch_index}\n")
                
                reporter.write("*"*40)
                
                distributed_list.append(distributed_labels_dict)
                initial_cluster_size_list.append(initial_cluster_size)
                min_samples_list.append(min_samples)
                sil_score_list.append(sil_score)
                db_index_list.append(db_index)
                ch_index_list.append(ch_index)

                if len(cluster_labels) == len(df_summary):
                    df_summary["cluster_id"] = cluster_labels
                    df_summary.to_csv(os.path.join(outputdir, f"hdbscan_clusterSize{initial_cluster_size}_min_samples{min_samples}_{content}.csv"), index=False)
        
        df = pd.DataFrame({
            "initial_cluster_size": initial_cluster_size_list,
            "min_samples": min_samples_list,
            "labels_distribution": distributed_list,
            "sil_score": sil_score_list,
            "db_index": db_index_list,
            "ch_index": ch_index_list
        })
        
        # sort dataframe
        df = df.sort_values(by=["sil_score", "db_index", "ch_index"], ascending=[False, True, False])
        print(f"the best initial cluster size and min_samples are {df.iloc[0]}")
        df.to_csv(os.path.join(outputdir, f"hdbscan_eval_{content}.csv"), index=False)
                
        reporter.close()
        
    elif mode =="clustering":
        # clustering using the best initial cluster size and min_samples.
        outputdir = os.path.join(data_path, f"clustering_{content}")    
        os.makedirs(outputdir, exist_ok=True)
        
        reporter = open(os.path.join(outputdir, f"cluster_report_{content}.csv"), "w+")
        # TODO: sample initial cluster size and min_samples.
        print(f" initializing the clsutering of {content} embeddings"+
             f" with min_cluster_size {min_clusters_size} and min_samples {min_samples_size}")
        hdbscan_labels, hdbscan_cluster_dict, initial_sil_score, initial_db_index, initial_ch_index = iterative_clustering(
            reporter,
            preprocessed_embeddings=preprocessed_embeddings,
            initial_cluster_size=min_clusters_size,
            min_samples=min_samples_size,)

        # iterative refinement
        print("refining clusters... using centroid")
        labels_centroid = refining_clustering(hdbscan_labels,
                                            hdbscan_cluster_dict,
                                            preprocessed_embeddings,
                                            initial_sil_score,
                                            initial_db_index,
                                            initial_ch_index, 
                                            reporter,
                                            "centroid")
        print("refining clusters... using centroid")
        labels_medoid = refining_clustering(hdbscan_labels,
                                            hdbscan_cluster_dict,
                                            preprocessed_embeddings,
                                            initial_sil_score,
                                            initial_db_index,
                                            initial_ch_index, 
                                            reporter,
                                            "medoid")

        # close the reporter
        reporter.close()
        
        outputfile = os.path.join(outputdir, f"{content}_clusters.csv")
        
        # write down the labels for clustering.
        print(f"saving the cluster labels to {outputfile}")
        if len(labels_centroid) == len(df_summary):
            df_summary["cluster_hdbscan"] = hdbscan_labels
            df_summary["cluster_id_centroid"] = labels_centroid
            df_summary["cluster_id_medoid"] = labels_medoid
            df_summary.to_csv(outputfile, index=False)

        labels_centroid_values = list(Counter(labels_centroid).values())
        labels_medoid_values = list(Counter(labels_medoid).values())
        print("Centroid labels, mean: ", np.mean(labels_centroid_values),
            "std: ", np.std(labels_centroid_values))
        print("medoid labels, mean: ", np.mean(labels_medoid_values),
            "std: ", np.std(labels_medoid_values))
    
    else: 
        print("choose mode in [exp, clustering]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clusterer script for refining clusters.")
    parser.add_argument("--data_path", type=str, default="data/summaries/gpt-4.1-nano",
                        help="Path to the directory with summaries and embeddings. ")
    parser.add_argument("--content", type=str, default="task",
                        help="What content embeddings should be considered [subject, task, summary]")
    parser.add_argument("--mode", type=str, default="exp",
                        help="In what mode to run the code [exp, clustering]")
    parser.add_argument("--min_clusters_size", type=int, default=150,
                        help="The initial cluster size to start with.")
    parser.add_argument("--min_samples_size", type=int, default=5,   
                        help="The initial min samples to start with.")
    args = parser.parse_args()

    # Pass the arguments to the main function
    main(data_path=args.data_path,
         content=args.content,
         mode = args.mode,
         min_clusters_size=args.min_clusters_size,
         min_samples_size=args.min_samples_size)    
