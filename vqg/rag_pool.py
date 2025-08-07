import json, random
from collections import defaultdict
import re

def stratified_split(data, db_ratio=0.5, seed=42):
    """
    split data into two sets based on scenario_name so that each scenario is represented in both sets.
    """
    random.seed(seed)

    buckets = defaultdict(list)
    for item in data:
        buckets[item["scenario_name"]].append(item)

    db_set, eval_set = [], []

    for name, vids in buckets.items():
        random.shuffle(vids)
        n_db = int(len(vids) * db_ratio)
        db_set.extend(vids[:n_db])
        eval_set.extend(vids[n_db:])

    return db_set, eval_set

#------------------

raw = json.load(open("log_formal/qa_val_samples_video_w_desc_test.json"))   # raw data
cleaned_annotations = []
for ann in raw:
    ann['annotations'] = [
        item for item in ann.get('annotations', [])
        if 'video' in item and 'A_hat' in item
    ]
    if ann['annotations']:
        cleaned_annotations.append(ann)
raw = cleaned_annotations
db_videos, eval_videos = stratified_split(raw, db_ratio=0.5)

print(f"DB videos : {len(db_videos)}")
print(f"Eval videos: {len(eval_videos)}")

with open("log_formal/qa_val_samples_video_w_desc_db.json", "w") as f:
    json.dump(db_videos, f, indent=4)
with open("log_formal/qa_val_samples_video_w_desc_eval.json", "w") as f:
    json.dump(eval_videos, f, indent=4)

#----------------above is to split the data----------------

#----------------below is to prepare the segments for FAISS index----------------
def flatten_segments(video_list):
    segments = []
    for v in video_list:
        for ann in v["annotations"]:
            for c in ann.get("A_hat", []):
                segments.append({
                    "id"        : f"{v['video_id']}#{ann['timestamp']}  ",
                    "text"      : re.sub(r'^\[(?:Good Execution|Tip for Improvement)\]', '', c).strip(),
                    "timestamp" : ann["timestamp"],
                    "video_path": ann["video"],
                    "scenario"  : v["scenario_name"]
                })
    return segments

db_segments = flatten_segments(db_videos)   # for faiss index

# build_faiss.py

import faiss, torch, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import re

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_JSON     = "db_segments_zsst.json"
IDX_OUT     = "log_formal/comment_db_zsst.index"
TXT_OUT     = "log_formal/comment_texts_zsst.json"

data = db_segments
texts = [c["text"] for c in data]

model = SentenceTransformer(EMBED_MODEL, device="cuda")
vecs  = model.encode(texts, convert_to_numpy=True, batch_size=128, show_progress_bar=True)
faiss.normalize_L2(vecs)

index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, IDX_OUT)
json.dump(data, open(TXT_OUT, "w"))
print("✅ FAISS & metadata saved.")