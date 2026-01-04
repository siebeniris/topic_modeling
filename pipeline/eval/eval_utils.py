import os 
import jsonlines 
from collections import defaultdict
import re

from pipeline.utils.embedding_utils import loading_embeddings
from pipeline.utils.utils import load_csv_file


def extract_response(text):
    """
    Extracts "no", "yes", or "na" (case-insensitive) from a string.

    Args:
        text: The input string.

    Returns:
        The extracted response (lowercase), or None if not found.
    """
    pattern = r"(?i)\b(no|yes|na)\b"  # Case-insensitive match for whole words
    match = re.search(pattern, text)

    if match:
        return match.group(1).lower()  # Return lowercase version
    else:
        return None
    
    
def load_summarization(datapath, content_type):
    if content_type == "outputs":
        filepath = os.path.join(datapath, "xxx.jsonl")
        conversation_dict = defaultdict(dict)
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                conv_id = obj["conversation_id"]
                turn_id = obj["turn_id"]                    
                conversation_dict[conv_id][turn_id] = {
                        "turn": obj["turn"], 
                        "embed_id": obj["embed_id"],
                        "input_tokens": obj["input_tokens"],
                        "output_tokens": obj["output_tokens"],
                        "summary": obj["summary"]
                    }
                
            
        return conversation_dict 
    elif content_type == "task_embeds":
        
        filepath = os.path.join(datapath, "embeddings_mpnet_task.zarr")
        return loading_embeddings(filepath)
    elif content_type == "task_meta":
        filepath = os.path.join(datapath, "embeddings_mpnet_task.csv")
        return load_csv_file(filepath, "task")
    
    elif content_type == "subject_embeds":
        
        filepath = os.path.join(datapath, "embeddings_mpnet_subject.zarr")
        return loading_embeddings(filepath)
    elif content_type == "subject_meta":
        filepath = os.path.join(datapath, "embeddings_mpnet_subject.csv")
        return load_csv_file(filepath, "task")
    
    elif content_type == "summary_embeds":
        
        filepath = os.path.join(datapath, "embeddings_mpnet_summary.zarr")
        return loading_embeddings(filepath)
    elif content_type == "summary_meta":
        filepath = os.path.join(datapath, "embeddings_mpnet_summary.csv")
        return load_csv_file(filepath, "summary")
    
    
    
    
def get_stats_system_message(system_message_exist_llm, system_message_exist_gold):
    sm_stats = {
        "No_System_Message": {
            "na":0, "yes":0, "no":0, -1:0
        },
        "System_Message": {
            "na":0, "yes":0, "no":0, -1:0
        }
    }
    for sm_llm, sm_gold in zip(system_message_exist_llm, system_message_exist_gold):
        if sm_gold==False:
            if sm_llm == "na":
                sm_stats["No_System_Message"]["na"] += 1
            else:
                if sm_llm == "yes":
                    sm_stats["No_System_Message"]["yes"] +=1
                elif sm_llm == "no":
                    sm_stats["No_System_Message"]["no"] +=1
                else:
                    sm_stats["No_System_Message"][-1] +=1
        elif sm_gold == True:
            if sm_llm == "na":
                sm_stats["System_Message"]["na"] += 1
            else:
                if sm_llm == "yes":
                    sm_stats["System_Message"]["yes"] +=1
                elif sm_llm == "no":
                    sm_stats["System_Message"]["no"] +=1
                else:
                    sm_stats["System_Message"][-1] +=1
    return sm_stats

def get_conversation_continuation(previous_conversation_exist_llm, previous_conversation_exist_gold):
    sm_stats = {
        "No_Previous_Conversation": {
            "na":0, "yes":0, "no":0, -1:0
        },
        "Previous_Conversation": {
            "na":0, "yes":0, "no":0, -1:0
        }
    }
    for sm_llm, sm_gold in zip(previous_conversation_exist_llm, previous_conversation_exist_gold):
        if sm_gold==False:
            if sm_llm == "na":
                sm_stats["No_Previous_Conversation"]["na"] += 1
            else:
                if sm_llm == "yes":
                    sm_stats["No_Previous_Conversation"]["yes"] +=1
                elif sm_llm == "no":
                    sm_stats["No_Previous_Conversation"]["no"] +=1
                else:
                    sm_stats["No_Previous_Conversation"][-1] +=1
        elif sm_gold == True:
            if sm_llm == "na":
                sm_stats["Previous_Conversation"]["na"] += 1
            else:
                if sm_llm == "yes":
                    sm_stats["Previous_Conversation"]["yes"] +=1
                elif sm_llm == "no":
                    sm_stats["Previous_Conversation"]["no"] +=1
                else:
                    sm_stats["Previous_Conversation"][-1] +=1
    return sm_stats
