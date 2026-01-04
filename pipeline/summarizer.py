import os
import sys
import json
import threading
import argparse
from typing import Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm
import glob

from sentence_transformers import SentenceTransformer
from langsmith import wrappers
from langchain_ollama import OllamaLLM
from anthropic import Anthropic
from openai import OpenAI

from langchain_core.prompts import PromptTemplate
from datetime import datetime

from pipeline.utils.prompts_fewshot import convo_prompt
from pipeline.utils.summarization_utils import (parse_summarization_output_updated,
                                                save_or_load_examples,
                                                is_default_message,
                                                found_the_last_conversation_id,
                                                find_position_and_slice)
from pipeline.utils.embedding_utils import save_embeddings_to_zarr

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))


env_loaded = load_dotenv(".env")


file_lock = threading.Lock()


def write_to_jsonl(output_file: str, result: Dict[str, Any]):
    """Write a single result to the JSONL file with thread safety."""
    with file_lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')


def load_data_and_llm(model_name="gpt-4o-mini",
                      data_name="xxx"):

    assert model_name in ["gpt-4o-mini",
                          "gpt-4.1-nano", "claude-3-5-haiku-20241022"]

    # load the dataset either locally or from langsmith
    conversation_lists = save_or_load_examples(data_name)
    print(
        f"Loading data from {data_name} with {len(conversation_lists)} conversations")

    # initialize the environmental keys.
    api_key = os.environ.get("xxx")

    print(f"loading model {model_name}...")

    if "claude" in model_name:
        anthropic_base_url = os.environ.get("anthropic_base_url")
        llm_client = wrappers.wrap_anthropic(Anthropic(
            base_url=anthropic_base_url,
            auth_token=api_key

        ))
    elif "gpt" in model_name:
        openai_base_url = os.environ.get("openai_base_url")
        llm_client = wrappers.wrap_openai(OpenAI(
            api_key=api_key,
            base_url=openai_base_url,

        ))
    else:
        print("Choose OpenAI or Anthropic models...")
        llm_client = None

    return llm_client, conversation_lists


def processing_one_conversation(convo,
                                llm_client,
                                model_name,
                                convo_prompt_formatted,
                                sentence_transformer,
                                outputfile,
                                task_embeds_file,
                                task_meta_writer,
                                subject_embeds_file,
                                subject_meta_writer,
                                summary_embeds_file,
                                summary_meta_writer,
                                last_conversation_id,
                                last_turn_id,
                                previous_convo_threshold=4):
    convo_id = convo[0]  # the conversation id
    # including "System Message", "Human:.. AI:..." conversation turns.
    turns = convo[1]

    # control variables for summarizing one conversation.
    system_message_summary = None
    previous_conversation_text = None
    previous_convo_nr = 0

    task_list = []
    subject_list = []
    summary_list = []
    # processing the conversation turns
    for turn_id, turn in tqdm(enumerate(turns)):
        # if it is a system message, we need to call the system message prompt
        if turn.startswith("System Message:") and not is_default_message(turn):
            system_message_summary = turn
        else:
            try:
                # a conversation turn
                if last_conversation_id and last_turn_id:
                    if convo_id == last_conversation_id and turn_id <= last_turn_id:
                        continue

                if "gpt" in model_name:
                    turn_response = llm_client.chat.completions.create(
                        model=model_name,
                        max_completion_tokens=500,
                        messages=[{
                            "role": "user",
                            "content": convo_prompt_formatted.format(message=system_message_summary,
                                                                    previous_conversation=previous_conversation_text,
                                                                    input=turn)}])
                    turn_output = turn_response.choices[0].message.content
                    input_tokens_nr = turn_response.usage.prompt_tokens
                    output_tokens_nr = turn_response.usage.completion_tokens

                elif "claude" in model_name:
                    turn_response = llm_client.messages.create(
                        model=model_name,
                        max_tokens=500,
                        messages=[{
                            "role": "user",
                            "content": convo_prompt_formatted.format(message=system_message_summary,
                                                                    previous_conversation=previous_conversation_text,
                                                                    input=turn)}])
                    turn_output = turn_response.content[0].text
                    input_tokens_nr = turn_response.usage.input_tokens
                    output_tokens_nr = turn_response.usage.output_tokens

            
                # dictionary
                turn_summary = parse_summarization_output_updated(turn_output)
                embed_id = f"{convo_id}_{turn_id}"
                # embeddings,
                task = turn_summary["Task"]
                subject = turn_summary["Subject"]
                summary = turn_summary["Summary"]

                task_list.append(task)
                subject_list.append(subject)
                summary_list.append(summary)

                task_meta_writer.write(f"{convo_id},{turn_id},{task}\n")
                subject_meta_writer.write(f"{convo_id},{turn_id},{subject}\n")
                summary_meta_writer.write(f"{convo_id},{turn_id},{summary}\n")

            except Exception as e:

                turn_summary = None
                embed_id = None
                turn_output = None
                print(
                    f"Error parsing turn output: {e}, the output {turn_output}")

            # print(f'system message: {system_message_summary}')
            # print(
            #     f'previous {previous_convo_nr} convos : {previous_conversation_text}')
            # print(f"current convo: {turn}")
            # print("-"*40)
            # print(turn_summary)
            # print(turn_output)

            # turn accumulation
            if previous_conversation_text is None:
                previous_conversation_text = turn
                previous_convo_nr += 1
            else:
                # see previous conversation turns up until a threshold
                if previous_convo_nr < previous_convo_threshold:
                    previous_conversation_text += "\n" + turn
                    previous_convo_nr += 1
                else:
                    previous_conversation_text = turn
                    previous_convo_nr = 1

            # output
            if turn_output:
                output_dict = {
                    "conversation_id": convo_id,
                    "turn_id": turn_id,
                    "turn": turn,
                    'embed_id': embed_id,
                    "input_tokens": input_tokens_nr,
                    "output_tokens": output_tokens_nr,
                    "turn_output": turn_output,
                    "summary": turn_summary
                }
                write_to_jsonl(outputfile, output_dict)

            # save embeddings.

    if task_list:
        task_embeds = sentence_transformer.encode(task_list)
        subject_embeds = sentence_transformer.encode(subject_list)
        summary_embeds = sentence_transformer.encode(summary_list)

        # print(task_embeds.shape)

        save_embeddings_to_zarr(task_embeds, task_embeds_file)
        save_embeddings_to_zarr(subject_embeds, subject_embeds_file)
        save_embeddings_to_zarr(summary_embeds, summary_embeds_file)


def main(model_name="gpt-4o-mini",
         data_name="xx",
         outputdir="outputs",
         mode='resume',
         samples=-1):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputdir = os.path.join(outputdir, data_name)

    # load the most recent outputdir, starting with the model_name
    if mode == 'resume':
        model_dirs = glob.glob(os.path.join(
            outputdir, f"{model_name.replace('/', '_')}_*"))
        if model_dirs:
            # Sort directories by timestamp (extracted from the folder name)
            model_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            model_outputdir = model_dirs[0]
            print(
                f"Resuming from the latest output directory: {model_outputdir}")
        else:
            print(
                f"No existing output directory found for model {model_name}. Starting fresh.")
            model_outputdir = os.path.join(
                outputdir, f"{model_name.replace('/', '_')}_{timestamp}")
            os.makedirs(model_outputdir, exist_ok=True)
    else:
        # Create a new output directory
        model_outputdir = os.path.join(
            outputdir, f"{model_name.replace('/', '_')}_{timestamp}")
        os.makedirs(model_outputdir, exist_ok=True)

    # intialize sentence transformer
    sentence_transformer = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2")

    # initialize writer files.
    task_embeds_file = os.path.join(
        model_outputdir, "embeddings_mpnet_task.zarr")
    task_embeds_meta_file = os.path.join(
        model_outputdir, "embeddings_mpnet_task.csv")

    subject_embeds_file = os.path.join(
        model_outputdir, "embeddings_mpnet_subject.zarr")
    subject_embeds_meta_file = os.path.join(
        model_outputdir, "embeddings_mpnet_subject.csv")

    summary_embeds_file = os.path.join(
        model_outputdir, "embeddings_mpnet_summary.zarr")
    summary_embeds_meta_file = os.path.join(
        model_outputdir, "embeddings_mpnet_summary.csv")

    # outputfile to save the summary output
    outputfile = os.path.join(model_outputdir, f"{data_name}.jsonl")

    print(f"Loading {model_name} and data {data_name}")
    # load the model and dataset
    llm_client, conversations = load_data_and_llm(model_name=model_name,
                                                  data_name=data_name)

    # formatting the prompt.
    convo_prompt_formatted = PromptTemplate(
        template=convo_prompt, input_variables=[
            "message", "previous_conversation", "input"],
    )

    # find the conversation.
    if mode == "resume":
        # look at the outputfile's last line.
        last_conversation_id, last_turn_id = found_the_last_conversation_id(
            outputfile)
        print(
            f"found the last processed conversation-turn {last_conversation_id}-{last_turn_id}")
        conversations = find_position_and_slice(
            conversations, last_conversation_id)
        print(f"There are {len(conversations)} conversations to process...")
    else:
        last_conversation_id = None
        last_turn_id = None

    # open the file writers.
    task_meta_writer = open(task_embeds_meta_file, "a+")
    subject_meta_writer = open(subject_embeds_meta_file, "a+")
    summary_meta_writer = open(summary_embeds_meta_file, "a+")

    if mode != "resume":
        task_meta_writer.write(f"convo_id,turn_id,task\n")
        subject_meta_writer.write(f"convo_id,turn_id,subject\n")
        summary_meta_writer.write(f"convo_id,turn_id,summary\n")

    if samples > 0:
        conversations = conversations[:samples]
        print(f"processing {samples} conversations..")

    print(f"outputing summaries to {outputfile}")
    for conversation in tqdm(conversations):
        processing_one_conversation(conversation,
                                    llm_client,
                                    model_name,
                                    convo_prompt_formatted,
                                    sentence_transformer,
                                    outputfile,
                                    task_embeds_file,
                                    task_meta_writer,
                                    subject_embeds_file,
                                    subject_meta_writer,
                                    summary_embeds_file,
                                    summary_meta_writer,
                                    last_conversation_id=last_conversation_id,
                                    last_turn_id=last_turn_id,
                                    previous_convo_threshold=4)


    subject_meta_writer.close()
    task_meta_writer.close()
    summary_meta_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarizer script for processing conversations.")
    parser.add_argument("--model_name", type=str,
                        default="gpt-4o-mini", help="Name of the model to use.")
    parser.add_argument("--data_name", type=str, default="xxx",
                        help="Name of the dataset to process.")
    parser.add_argument("--outputdir", type=str, default="data/summaries/",
                        help="Directory to save the output summaries.")
    parser.add_argument("--samples", type=int, default=-1, help="Samples to process...")
    parser.add_argument("--mode", type=str, default="resume",
                        help="Samples to process...")

    args = parser.parse_args()

    main(model_name=args.model_name,
         data_name=args.data_name,
         outputdir=args.outputdir,
         mode=args.mode,
         samples=args.samples)
