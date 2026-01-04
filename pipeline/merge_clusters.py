import os 

import pandas as pd
from langsmith import Client, wrappers
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from pipeline.utils.prompts_merge_clusters import template_merge_clusters_examples, template_updated, higher_level_clusters_non_overlapping


# deduplicate across clusters
env_loaded = load_dotenv(".env")
api_key = os.environ.get("xxx")
openai_base_url = os.environ.get("openai_base_url")



ls_client = Client()
openai_client = wrappers.wrap_openai(OpenAI(
    api_key= api_key,
    base_url= openai_base_url
    ))
model_name = "gpt-4o-mini"
# tokenizer = tiktoken.get_encoding("o200k_base")

def load_cluster_name_summary(content, threshold, samples_nr=10):
    filepath = f"data/gpt-4o-mini_20250508_161651/clustering_{content}/{content}_base_cluster_names_th{threshold}.csv"
    df = pd.read_csv(filepath).sort_values(by="cluster_id", ascending=True)
    text_string = ""
    for cluster_id, name, summary, samples in zip(df["cluster_id"], df["name"], df["summary"], df["samples"]):
        text_string += f"\nCluster {cluster_id}. {name}:{summary}\n"
        text_string += f"Examples for this cluster:\n"
        text_string += "\n".join(samples.split("\n")[:samples_nr])
    return text_string
    # try: 
    #     num_tokens = len(tokenizer.encode(text_string))
    #     if num_tokens <= 91659:
    #         print("The number of tokens {num_tokens} are within limits..")
    #         return text_string
    # except Exception:
    #         print("The number of tokens {num_tokens} are out of limits..")
    #         return None 
    


def main(content, threshold, template="updated",
         samples = 10, 
         low_bound_desired_number=10,
         upper_bound_desired_number=20,
         desired_number=15):
    print(f"loading {content} with th {threshold} and {samples} incluster-samples")
    text_string = load_cluster_name_summary(content, threshold, samples_nr=samples)
    outputdir = f"data/gpt-4o-mini_20250508_161651/clustering_{content}"
    outputdir = os.path.join(outputdir, "higher_level_clusters")
    os.makedirs(outputdir, exist_ok=True)
    
    output_filename = f"{content}_th{threshold}_samples{samples}_"
    if template=="normal":
        merge_cluster_prompt_formatted = PromptTemplate(
        template=template_updated,
        input_variables=["low_bound_desired_number", "upper_bound_desired_number", "desired_number", "cluster_list"]
        )
        output_filename+="normal"
    elif template=="non-overlapping":
        merge_cluster_prompt_formatted = PromptTemplate(
        template=higher_level_clusters_non_overlapping,
        input_variables=["low_bound_desired_number", "upper_bound_desired_number", "desired_number", "cluster_list"]
        )
        output_filename+="non-overlapping"
    else:
        merge_cluster_prompt_formatted = PromptTemplate(
        template=template_merge_clusters_examples,
        input_variables=["low_bound_desired_number", "upper_bound_desired_number", "desired_number", "cluster_list"]
        )
        output_filename+="withExampels"
    
    higher_level_cluster_name_formatted = merge_cluster_prompt_formatted.format(
    low_bound_desired_number = low_bound_desired_number, 
    upper_bound_desired_number = upper_bound_desired_number,
    desired_number=desired_number,
    cluster_list=text_string
    )

    response = openai_client.chat.completions.create(
        model= model_name,
        messages=[{
            "role":"user",
            "content": higher_level_cluster_name_formatted
        }]
    )

    response_output = response.choices[0].message.content
    input_tokens_nr = response.usage.prompt_tokens
    output_tokens_nr = response.usage.completion_tokens
    print(f"input tokens {input_tokens_nr}, output tokens {output_tokens_nr}")
    try: 
        if input_tokens_nr <= 91659:
            print("The number of tokens {num_tokens} are within limits..")
            output_filename+=".txt"
            print(f"saving output to {os.path.join(outputdir, output_filename)}...")
            with open(os.path.join(outputdir, output_filename),"w")as f:
                f.write(response_output)
                
    except Exception:
            print("The number of tokens {num_tokens} are out of limits..")
            return None 
    print("*"*40)    
    
        

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Naming the clusters.")
    # parser.add_argument("--content", type=str,
    #                     default="subject", help="subject or content")
    # parser.add_argument("--threshold", type=int,
    #                     default=50, help="50, 100, 200")
    # parser.add_argument("--template", type=str,
    #                     default="normal", help="normal or examples")
    # parser.add_argument("--samples", type=int,
    #                     default=10, help="normal or examples")
    
    # args = parser.parse_args()

    for c in ["subject", "task"]:
        for t in [50, 100, 200]:
            for p in ["non-overlapping"]:
                # for samples in [10,20,30,50,100]:
                for samples in [30, 50, 100]:
                    main(content=c, threshold=t, template=p, samples =samples )

    # main(content=args.content, threshold=args.threshold)
    
        




