import os
import numpy as np
import pandas as pd 

from tqdm import tqdm
from zadu.measures import *


os.getcwd()

from pipeline.utils.embedding_utils import (loading_embeddings,
                                            save_embeddings_to_zarr,
                                            preprocessing_embeddings,
                                            eval_cluster,
                                            caclualte_medoid_centroid_for_clusters
                                            )
from pipeline.utils.utils import load_csv_file
from pipeline.utils.summarization_utils import save_or_load_examples 

from langchain_core.prompts import PromptTemplate
from datetime import datetime

from pipeline.utils.prompts_fewshot import convo_prompt
from pipeline.summarizer import load_data_and_llm


from sklearn.preprocessing import StandardScaler
import umap
import pickle
from sentence_transformers import SentenceTransformer
import zarr


outputdir = "output/"
database_dir = "data/database"
os.makedirs(outputdir, exist_ok=True)

# all the data except annotated.
train_data_path = "data/database/train_task_data.csv"
# annotated data
test_data_path = "data/database/test_task_data.csv"


filepath_train_embeds = train_data_path.replace(".csv", ".zarr")
filepath_test_embeds = test_data_path.replace(".csv", ".zarr")

df_train = pd.read_csv(train_data_path)
df_test = pd.read_csv(test_data_path)

# loading sentence transformer
sentence_transformer = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2")

def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def extract_and_save_embeddings(tasks,filepath_tasks_embeds):
    iterator_list = chunk_list(tasks, 100)
    for l in tqdm(iterator_list):
        try: 
            task_embeds_l = sentence_transformer.encode(l)
            save_embeddings_to_zarr(task_embeds_l, filepath_tasks_embeds)
        except Exception as e:
            print(f"Error processing batch: {e}")
            print(l)



train_tasks = df_train["task"].tolist()
# post processed.
test_tasks = df_test["task_modified"].tolist()

extract_and_save_embeddings(train_tasks, filepath_train_embeds)
extract_and_save_embeddings(test_tasks, filepath_test_embeds)


# loading the whole embeddings to process.
embeddings_test = loading_embeddings(filepath_test_embeds)
embeddings_train = loading_embeddings(filepath_train_embeds)

scaler = StandardScaler()
scaler = scaler.fit(embeddings_train)
scaled_embeddings_train = scaler.transform(embeddings_train)

# the hyperparameters are set after comparing with PCA.
umap_reducer = umap.UMAP(n_neighbors=50, random_state=42, n_components=50, 
                            metric="euclidean")

trans = umap_reducer.fit(scaled_embeddings_train)
processed_embeddings_train = trans.embedding_
print(f" embeddings train processed : {processed_embeddings_train.shape}")


# saving embeddings .
save_embeddings_to_zarr(processed_embeddings_train, "data/database/processed_train_embeddings.zarr")

# svaing scaler and umap reducer.
with open(f"{outputdir}/scaler_task_embeddings.pkl", "wb") as f:
    pickle.dump(scaler, f)
    
with open(f"{outputdir}/umap_reducer_task_embeddings.pkl" , "wb") as f:
    pickle.dump(trans, f)


# processing the test embeddings
scaled_test_embeddings = scaler.transform(embeddings_test)
processed_test_embeddings = trans.transform(scaled_test_embeddings)

save_embeddings_to_zarr(processed_test_embeddings, "data/database/processed_test_embeddings.zarr")

# create train and test data for subclusters
y_train = df_train["subcluster_id"].to_numpy()
X_train = processed_embeddings_train
y_test = df_test["subcluster_id"].to_numpy()
X_test = processed_test_embeddings

X_subcluster_train_filename = 'data/database/knn_data/sub_cluster/X_train.npy'
X_subcluster_test_filename = 'data/database/knn_data/sub_cluster/X_test.npy'
y_subcluster_train_filename = 'data/database/knn_data/sub_cluster/y_train.npy'
y_subcluster_test_filename = 'data/database/knn_data/sub_cluster/y_test.npy'

np.save(X_subcluster_train_filename, X_train)
np.save(X_subcluster_test_filename, X_test)
np.save(y_subcluster_train_filename, y_train)
np.save(y_subcluster_test_filename, y_test)

assert X_train.shape[0]==y_train.shape[0]
assert X_test.shape[0]==y_test.shape[0]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# create train and test data for higher_level clusters 
y_train = df_train["higher_cluster_id"].to_numpy()
X_train = processed_embeddings_train
y_test = df_test["higher_cluster_id"].to_numpy()
X_test = processed_test_embeddings

assert X_train.shape[0]==y_train.shape[0]
assert X_test.shape[0]==y_test.shape[0]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


os.makedirs("data/database/knn_data/higher_level_cluster", exist_ok=True)
X_train_filename = 'data/database/knn_data/higher_level_cluster/X_train.npy'
X_test_filename = 'data/database/knn_data/higher_level_cluster/X_test.npy'
y_train_filename = 'data/database/knn_data/higher_level_cluster/y_train.npy'
y_test_filename = 'data/database/knn_data/higher_level_cluster/y_test.npy'

# 2. Use numpy.save() to save the arrays to files
np.save(X_train_filename, X_train)
np.save(X_test_filename, X_test)
np.save(y_train_filename, y_train)
np.save(y_test_filename, y_test)


