import os
import re
from typing import Tuple, Dict, List

import json
import jsonlines

from tqdm import tqdm
from langsmith import Client
from joblib import Parallel, delayed
from langsmith import wrappers


def processing_one_example(example) -> Tuple[str, list]:
    """Restructure conversations into conversation turns."""
    try:
        conversation_turns = []
        if "system_message" in example.inputs and example.inputs["system_message"] is not None:
            message = example.inputs["system_message"]
            if not is_default_message(message):
                conversation_turns.append(f"System Message: {message}")

        if "output" in example.inputs:
            human_output = None
            ai_output = "AI: "

            for line in example.inputs["output"]:
                if human_output is None and line["type"] == "human":
                    human_output = "Human: " + line["content"].strip()
                elif human_output:
                    if line["type"] == "ai":
                        ai_output += line["content"].strip()
                    elif line["type"] == "human":
                        convo = human_output + '\n' + ai_output
                        conversation_turns.append(convo)
                        human_output = "Human: " + line["content"].strip()
                        ai_output = "AI: "

            if human_output and ai_output != "AI: ":
                convo = human_output + '\n' + ai_output
                conversation_turns.append(convo)

        print(f"{example.id}: conversation turns {len(conversation_turns)}")
        return str(example.id), conversation_turns
    
    except Exception as e:
        print(f"Error processing example {example.id}: {e}")
        


def save_or_load_examples(dataset_name: str) -> List:
    """
    Processing and save or load examples from LangSmith dataset.
    """
    
    dataset_filepath = os.path.join("data", f"{dataset_name}.jsonl")

    if os.path.exists(dataset_filepath):
        print(f"loading conversation list {dataset_filepath}")
        conversation_lists = []
        with jsonlines.open(dataset_filepath, "r") as f:
            for obj in f:
                conversation_lists.append(obj)
        return conversation_lists
    else:
        print("Processing the examples from langsmith....")
        ls_client = Client()
        dataset = ls_client.read_dataset(dataset_name=dataset_name)
        print(f"loading examples ...")
        examples = ls_client.list_examples(dataset_id=dataset.id)
        conversation_lists = Parallel(
            n_jobs=-1)(delayed(processing_one_example)(example) for example in tqdm(examples))
        print(f"saving conversation to {dataset_filepath}")
        with jsonlines.open(dataset_filepath, "w") as f:
            for obj in conversation_lists:
                f.write(obj)
        return conversation_lists


def is_default_message(sentence: str) -> bool:
    """Check if the sentence matches the default system message pattern."""
    pattern = r"^Today's date is \d{4}-\d{2}-\d{2} You are a helpful assistant."
    return bool(re.fullmatch(pattern, sentence.strip()))


def parse_summarization_output_updated(output: str):
    """
    Parse the LLM summarization output, handling potential variations in formatting,
    ignoring any "Analysis" prefix.
    """
    parsed_output = dict()

    for line in re.split("\n+", output):
        line = line.strip()

        def extract_value(line, key):
            # Updated regex to handle *, **, and no * around keys, and extra spaces
            match = re.match(r"\s*(\*\*|\*)?\s*" + re.escape(key) + r"\s*:(\*\*|\*)?\s*(.*)", line, re.IGNORECASE)
            if match:
                return match.group(3).strip()
            return None


        task = extract_value(line, "Task")
        if task:
            parsed_output["Task"] = task

        subject = extract_value(line, "Subject")
        if subject:
            parsed_output["Subject"] = subject

        task_keyword = extract_value(line, "Task Keyword")
        if task_keyword:
            parsed_output["Task_Keyword"] = task_keyword

        subject_keyword = extract_value(line, "Subject Keyword")
        if subject_keyword:
            parsed_output["Subject_Keyword"] = subject_keyword

        language = extract_value(line, "Language")
        if language:
            parsed_output["Language"] = language

        translation_direction = extract_value(line, "Translation Direction")
        if translation_direction:
            parsed_output["Translation_Direction"] = translation_direction

        concerning_flag = extract_value(line, "Concerning")
        if concerning_flag:
            parsed_output["Concerning"] = concerning_flag

        summary = extract_value(line, "Summary")
        if summary:
            parsed_output["Summary"] = summary

        relevance = extract_value(line, "Relevance")
        if relevance:
            parsed_output["Relevance"] = relevance

        continua = extract_value(line, "Conversation Continuation")
        if continua:
            parsed_output["Conversation_Continuation"] = continua

    return parsed_output


def parse_summarization_output(output: str) -> Dict:
    """
    Parse the LLM summarization output with the prompt from promts_fewshot.py
    """
    parsed_output = dict()
    for line in re.split("\n+", output):
        line = line.strip()
        if re.match(r"\s*Task:", line):
            task = line[len(re.match(r"\s*Task:", line).group(0)):].strip()
            parsed_output["Task"] = task
        elif re.match(r"\s*Subject:", line):
            subject = line[len(
                re.match(r"\s*Subject:", line).group(0)):].strip()
            parsed_output["Subject"] = subject
        elif re.match(r"\s*Task Keyword:", line):
            task_keyword = line[len(
                re.match(r"\s*Task Keyword:", line).group(0)):].strip()
            parsed_output["Task_Keyword"] = task_keyword
        elif re.match(r"\s*Subject Keyword:", line):
            subject_keyword = line[len(
                re.match(r"\s*Subject Keyword:", line).group(0)):].strip()
            parsed_output["Subject_Keyword"] = subject_keyword
        elif re.match(r"\s*Language:", line):
            language = line[len(
                re.match(r"\s*Language:", line).group(0)):].strip()
            parsed_output["Language"] = language
        elif re.match(r"\s*Translation Direction:", line):
            translation_direction = line[len(
                re.match(r"\s*Translation Direction:", line).group(0)):].strip()
            parsed_output["Translation_Direction"] = translation_direction
        elif re.match(r"\s*Concerning:", line):
            concerning_flag = line[len(
                re.match(r"\s*Concerning:", line).group(0)):].strip()
            parsed_output["Concerning"] = concerning_flag
        elif re.match(r"\s*Summary:", line):
            summary = line[len(
                re.match(r"\s*Summary:", line).group(0)):].strip()
            parsed_output["Summary"] = summary
        elif re.match(r"\s*Relevance:", line):
            relevance = line[len(
                re.match(r"\s*Relevance:", line).group(0)):].strip()
            parsed_output["Relevance"] = relevance
        elif re.match(r"\s*Conversation Continuation:", line):
            continua = line[len(
                re.match(r"\s*Conversation Continuation:", line).group(0)):].strip()
            parsed_output["Conversation_Continuation"] = continua
    return parsed_output


def found_the_last_conversation_id(filepath):
    """
    Read from the last line of the file, and find the last conversation_id and turn_id.

    Args:
        filepath (_type_): path to the outputfile.

    Returns:
        _type_: (conversation_id, turn_id)
    """
    with open(filepath, "rb") as f:
        f.seek(0, 2)  # seek from the end of the line
        position = f.tell()
        line = b""

        # read backward until a newline is found
        while position > 0:
            position -= 1
            f.seek(position)
            char = f.read(1)
            if char == b"\n" and line:
                break
            line = char + line

        # Decode the last line and parse it as JSON
        last_line = line.decode('utf-8').strip()
        last_entry = json.loads(last_line)
        conversation_id = last_entry.get("conversation_id")
        turn_id = last_entry.get("turn_id")
        return conversation_id, turn_id


def find_position_and_slice(conversations, target_conversation_id):
    """
    Find the position of the specified conversation_id in the list of tuples
    and return the list of tuples from that position onward.

    Args:
        conversations (list): List of tuples in the format (conversation_id, conversation_turns).
        target_conversation_id (str): The conversation_id to search for.

    Returns:
        list: Sliced list of tuples starting from the target conversation_id.
    """
    for index, (conversation_id, _) in enumerate(conversations):
        if conversation_id == target_conversation_id:
            # Return the sliced list from the found position
            return conversations[index:]
    return []  # Return an empty list if the conversation_id is not found

