import jsonlines 
import os 
import json 
import pandas as pd
from tqdm import tqdm
import re 
from collections import defaultdict, Counter


from pipeline.utils.summarization_utils import (save_or_load_examples, 
                                                is_default_message)
from pipeline.eval.eval_utils import (load_summarization, 
                                      extract_response,
                                      get_conversation_continuation, 
                                      get_stats_system_message)

################ evaluation ################
#### Sanity Checks
# 1. whether system message exists or not
# 2. whether previous conversation turns exist or not
#### Quality Evaluation
# 1. string matching ? results from different LLMs
# 2. embed(task+subject)~ embed(summary)


dataset_name = "xxx"
conversation_lists = save_or_load_examples(dataset_name)


gpt4o_mini_outputs_path ="data/summaries/gpt-4o-mini"
gpt41_nano_outputs_path = "data/summaries/gpt-4.1-nano"
claude_outputs_path = "data/summaries/claude-3-5-haiku"

gpt4o_mini_outputs = load_summarization(gpt4o_mini_outputs_path, "outputs")
gpt41_nano_outputs = load_summarization(gpt41_nano_outputs_path, "outputs")
claude_outputs = load_summarization(claude_outputs_path, "outputs")


##### 

system_message_exist_gold = []
previous_conversation_exist_gold = []
# -1, NA, No, Yes 
system_message_exist_gpt4o_mini = []
previous_conversation_exist_gpt4o_mini = []

system_message_exist_gpt41_nano= []
previous_conversation_exist_41_nano = []

system_message_exist_claude = []
previous_conversation_exist_claude = []


conversation_id_list = []
turn_id_list = []


for conversation in tqdm(conversation_lists):
    conversation_id, turns = conversation
    
    system_message_exist = False
    previous_conversation_exist = False
    for turn_id, turn in enumerate(turns):
        conversation_id_list.append(conversation_id)
        turn_id_list.append(turn_id)
        

        if turn.startswith("System Message:"):
            if not is_default_message(turn):
                system_message_exist = True
        else:
            # turn which are not system message, there is previous conversation
            if system_message_exist:
                if turn_id > 1:
                    # previous conversation exist
                    previous_conversation_exist = True
            else:
                if turn_id > 0:
                    previous_conversation_exist = True
                    
        system_message_exist_gold.append(system_message_exist)
        previous_conversation_exist_gold.append(previous_conversation_exist)
            
        def add_stats_(llm_outputs, 
                llm_system_messsage_exist, 
                llm_previous_conversation_exist):
            if conversation_id in llm_outputs and turn_id in llm_outputs[conversation_id]:
                if llm_outputs[conversation_id][turn_id]["summary"]:
                    if "Relevance" in llm_outputs[conversation_id][turn_id]["summary"]:
                        relevance_system_message = extract_response(llm_outputs[conversation_id][turn_id]["summary"]["Relevance"])
                        llm_system_messsage_exist.append(relevance_system_message)
                    else:
                        llm_system_messsage_exist.append(-1)
                    
                    
                    if "Conversation_Continuation" in llm_outputs[conversation_id][turn_id]["summary"]:
                        continuation_previous_conversation = extract_response(llm_outputs[conversation_id][turn_id]["summary"]["Conversation_Continuation"])
                        llm_previous_conversation_exist.append(continuation_previous_conversation)
                    else:
                        llm_previous_conversation_exist.append(-1)
                        
                else:
                    llm_system_messsage_exist.append(-1)
                    llm_previous_conversation_exist.append(-1)
            else:
                llm_system_messsage_exist.append(-1)
                llm_previous_conversation_exist.append(-1)
        
        
        add_stats_(gpt4o_mini_outputs,
                system_message_exist_gpt4o_mini,
                previous_conversation_exist_gpt4o_mini)
        
        add_stats_(gpt41_nano_outputs,
                system_message_exist_gpt41_nano,
                previous_conversation_exist_41_nano)
        
        add_stats_(claude_outputs,
                system_message_exist_claude,
                previous_conversation_exist_claude)


### stats:
gpt41_nano_system_message_stats = get_stats_system_message(system_message_exist_gpt41_nano, system_message_exist_gold)
gpt41_nano_conversation_continuation_stats = get_conversation_continuation(previous_conversation_exist_41_nano, previous_conversation_exist_gold)

gpt4o_mini_system_message_stats = get_stats_system_message(system_message_exist_gpt4o_mini, system_message_exist_gold)
gpt4o_mini_conversation_continuation_stats = get_conversation_continuation(previous_conversation_exist_gpt4o_mini, previous_conversation_exist_gold)

claude_system_message_stats = get_stats_system_message(system_message_exist_claude, system_message_exist_gold)
claude_conversation_continuation_stats = get_conversation_continuation(previous_conversation_exist_claude, previous_conversation_exist_gold)

print("stats")
print("********* GPT-4.1-Nano *********")
print(f"System Message Stats ")
print(gpt41_nano_system_message_stats)
print("Conversation Continuation Stats")
print(gpt41_nano_conversation_continuation_stats)

print("********* GPT-4o-Mini *********")
print(f"System Message Stats ")  
print(gpt4o_mini_system_message_stats)
print("Conversation Continuation Stats")
print(gpt4o_mini_conversation_continuation_stats)

print("********* Claude *********")
print(f"System Message Stats ")
print(claude_system_message_stats)
print("Conversation Continuation Stats")
print(claude_conversation_continuation_stats)
                
