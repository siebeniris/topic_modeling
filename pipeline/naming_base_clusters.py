import os 
import random
import itertools 
from collections import defaultdict
import re
import argparse

import pandas as pd
from langsmith import Client, wrappers
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from pipeline.utils.prompts_naming_clusters import naming_cluster_prompt
from pipeline.utils.embedding_utils import loading_embeddings


env_loaded = load_dotenv(".env")
api_key = os.environ.get("xxx")
openai_base_url = os.environ.get("openai_base_url")

# initialize the client
ls_client = Client()
openai_client = wrappers.wrap_openai(OpenAI(
    api_key= api_key,
    base_url= openai_base_url
    ))
model_name = "gpt-4o-mini"


def parse_output(output):
    summary_match = re.search(r"<summary>(.*?)<\/summary>", output, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()
        print("Summary:", summary)
    else:
        summary = None
        print("Summary not found")

    # Extract the name
    name_match = re.search(r"<name>(.*?)<\/name>", output, re.DOTALL)
    if name_match:
        name = name_match.group(1).strip()
        print("Name:", name)
    else:
        name = None
        print("Name not found")
    return summary, name


def naming_clusters(content, threshold=50):
    embedding_path = f"data/gpt-4o-mini_20250508_161651/embeddings_mpnet_{content}_processed.zarr"
    embeddings = loading_embeddings(embedding_path)
    
    df = pd.read_csv(f"data/gpt-4o-mini_20250508_161651/clustering_{content}/{content}_clusters_overall.csv")
    cluster_ids = df[f"overall_{content}_id"].to_numpy()
    contents = df[f"{content}"].to_numpy()
    
    
    print("Calculating the pairwise distance between embeddings...")
    dist_list = pdist(embeddings, metric="euclidean")
    distance_matrix = squareform(dist_list)
    
    # group indices by cluster
    clusters=defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        clusters[cluster_id].append(idx)
    print(f"there are {len(clusters)} clusters ...")
    
    # get random 50 or cluster size samples
    # positive samples.
    random_samples = {}
    for cluster_id, indices in clusters.items():
        random_samples[int(cluster_id)] = random.sample(indices, min(threshold, len(indices)))

        
    # contrastive negative sampels
    contrastive_examples = defaultdict(list)
    for cluster_id, sampled_indices in random_samples.items():
        for idx in sampled_indices:
            distances = distance_matrix[idx]
            # find indices of examples in other clusters
            other_cluster_indices= [i for i,cid in enumerate(cluster_ids) if cid !=cluster_id]

            # get distances to examples in other clusters
            other_distances = [(i, distances[i]) for i in other_cluster_indices]

            other_distances = sorted(other_distances, key=lambda x: x[1])
            closest_contrastive = [i for i, _ in other_distances[:threshold]]
            
            # Store the contrastive examples
            contrastive_examples[cluster_id].append(closest_contrastive)

    flatten_contrastive_samples = {cluster_id : random.sample(list(set(list(itertools.chain.from_iterable(samples)))), threshold) 
                               for cluster_id, samples in contrastive_examples.items()}

    
    # using random samples and contrastive
    # iteratively  put positive and negative samples in the prompt template.

    base_name_template= PromptTemplate(
        template=naming_cluster_prompt, 
        input_variables=["statements", "contrastive_statements"]
    )
    
    cluster_ids = []
    random_samples_list = []
    contrastive_samples_list = []
    response_list = []
    summary_list = []
    name_list = []
    used_tokens = []
    for cluster_id, random_sample_cluster_ids in tqdm(random_samples.items()):
        contrastive_example_cluster_ids = flatten_contrastive_samples[cluster_id]
        random_sample_cluster = contents[random_sample_cluster_ids]
        contrastive_examples_cluster = contents[contrastive_example_cluster_ids]

        print(f"Cluster {cluster_id}, Positive {len(random_sample_cluster)}, Contrastive {len(contrastive_examples_cluster)}")
        statements = "\n".join(random_sample_cluster)
        contrastive_statements = "\n".join(contrastive_examples_cluster)

        base_name_prompt_formatted = base_name_template.format(
            statements= statements,
            contrastive_statements = contrastive_statements
        )

        print(f"Prompting...")

        response = openai_client.chat.completions.create(
            model= model_name, 
            max_completion_tokens=200,
            messages= [{
                "role":"user",
                "content": base_name_prompt_formatted
            }])
        response_output = response.choices[0].message.content
        input_tokens_nr = response.usage.prompt_tokens
        output_tokens_nr = response.usage.completion_tokens
        summary, name = parse_output(response_output)
        print(f"prompt tokens {input_tokens_nr}, completion {output_tokens_nr}")
        used_tokens = input_tokens_nr + output_tokens_nr
        cluster_ids.append(cluster_id)
        random_samples_list.append(statements)
        contrastive_samples_list.append(contrastive_statements)
        response_list.append(response_output)
        summary_list.append(summary)
        name_list.append(name)

    df = pd.DataFrame({
        "cluster_id":cluster_ids, 
        "samples": random_samples_list,
        "contrastive_samples": contrastive_samples_list,
        "output":response_list,
        "summary": summary_list, 
        "name": name_list
    })
    
    outputfile = f"data/gpt-4o-mini_20250508_161651/clustering_{content}/{content}_base_cluster_names_th{threshold}.csv"
    df.to_csv(outputfile, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Naming the clusters.")
    parser.add_argument("--content", type=str,
                        default="subject", help="subject or content")
    parser.add_argument("--threshold", type=int,
                        default=50, help="subject or content")
    
    args = parser.parse_args()

    naming_clusters(content=args.content, threshold=args.threshold)
    