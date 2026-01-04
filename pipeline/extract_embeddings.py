import jsonlines 
import os 
import json 
import pandas as pd
from tqdm import tqdm
import re 
from collections import defaultdict, Counter
import evaluate 
import numpy as np
from sentence_transformers import SentenceTransformer


from pipeline.utils.summarization_utils import save_or_load_examples, is_default_message
from pipeline.utils.embedding_utils import loading_embeddings,save_embeddings_to_zarr
from pipeline.utils.utils import load_csv_file

df = pd.read_csv("data/xx.csv")

gpt_path = "data/summaries/gpt-4o-mini_20250508_161651"
save_path = "data/gpt-4o-mini_20250508_161651"

sentence_transformer = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2")

def chunk_list_generator(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]
        

def extract_embeddings(save_path, data_path, content_type="task"):
    meta_df = load_csv_file(f"{data_path}/embeddings_mpnet_{content_type}.csv", content=content_type)
    meta_df["turn_id"] = meta_df["turn_id"].astype(int)
    meta_df_filtered = meta_df.merge(df, left_on=["convo_id", "turn_id"], right_on=["convo_id", "turn_id"], how="inner")
    content_list = meta_df_filtered[content_type].tolist()
    # extract embedings 
    embeds_file = os.path.join(save_path, f"embeddings_mpnet_{content_type}.zarr")
    meta_df_filtered = meta_df_filtered[["convo_id", "turn_id", content_type]]
    meta_df_filtered.to_csv(os.path.join(save_path, f"embeddings_mpnet_{content_type}.csv"), index=False)
    print(f"Extracting {content_type} embeddings.. length {len(meta_df_filtered)}.")
    task_list = chunk_list_generator(content_list, 1000)
    for i, task_chunk in tqdm(enumerate(task_list)):
        task_embeds = sentence_transformer.encode(task_chunk)
        save_embeddings_to_zarr(task_embeds, embeds_file)


extract_embeddings(save_path, gpt_path, content_type="task")

extract_embeddings(save_path, gpt_path, content_type="subject")

extract_embeddings(save_path, gpt_path, content_type="summary")