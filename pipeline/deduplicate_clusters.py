import os 

from langsmith import Client, wrappers
from openai import OpenAI
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

from langchain_core.prompts import PromptTemplate
from pipeline.utils.prompts_cluster_deduplication import updated_deduplication_prompt


# deduplicate across clusters
env_loaded = load_dotenv(".env")
api_key = os.environ.get("xx")
openai_base_url = os.environ.get("openai_base_url")

# client.
ls_client = Client()
openai_client = wrappers.wrap_openai(OpenAI(
    api_key= api_key,
    base_url= openai_base_url
    ))
model_name = "gpt-4o-mini"


def load_clusters(filepath):
    with open(filepath) as f:
        return f.read()
    
def deduplicating_clusters(filepath):
    clusters = load_clusters(filepath)
    print(clusters)
    output_filename = os.path.basename(filepath)
    output_folder = "data/gpt-4o-mini/deduplicated_clusters"
    outputfile = os.path.join(output_folder, output_filename)
    
    deduplicate_prompt= PromptTemplate(
        template=updated_deduplication_prompt,
        input_variables=["input"]
    )
    
    deduplicate_prompt_formatted = deduplicate_prompt.format(
        input = clusters
    )
    
    response = openai_client.chat.completions.create(
        model= model_name,
        messages=[{
            "role":"user",
            "content": deduplicate_prompt_formatted
        }]
    )
    response_output = response.choices[0].message.content
    input_tokens_nr = response.usage.prompt_tokens
    output_tokens_nr = response.usage.completion_tokens
    print(f"input tokens {input_tokens_nr}, output tokens {output_tokens_nr}")
    
    print(f"saving the output to {outputfile}")
    with open(outputfile, "w") as f:
        f.write(response_output)


if __name__ == "__main__":
    import plac 
    plac.call(deduplicating_clusters)