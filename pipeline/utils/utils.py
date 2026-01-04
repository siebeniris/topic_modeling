import os
import pandas as pd


def load_csv_file(filepath:str, content:str) -> pd.DataFrame:
    # when the delimiter is restricted to the first two.
    convo_ids = []
    turn_ids = []
    summaries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if not line.startswith("convo_id,"):
                items = line.replace("\n", "").split(",", 2)
                if len(items)==3:
                    convo_id, turn_id, summary = items 
                    convo_ids.append(convo_id)
                    turn_ids.append(turn_id)
                    summaries.append(summary)
                else:
                    print(f"malformed row: {line}")
                    
    return pd.DataFrame({
        "convo_id": convo_ids,
        "turn_id":turn_ids,
        content: summaries
    })