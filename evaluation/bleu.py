# 1. 读取clip-comments数据和vlm生成的提问数据

# 2. 计算所有的comments和提问的embeddings

# 3. 统计每一个clip的评论数。如果clip数*6<15则取15，否则取clip数*6，以构筑检索池。并且，检索池里的条目（1）尽量从同一个视频里取（2）如果同一个视频不够，则从其他同一task的视频里取。

import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import random
import statistics
from sklearn.metrics.pairwise import cosine_similarity
import time
import re
import evaluate

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval evaluation")
    parser.add_argument("--filename", type=str, default=None, help="filename")
    args = parser.parse_args()
    return args
args = parse_args()

# ---------- 读取数据 ----------
file_name = "log_formal/"+args.filename+".json"
with open(file_name, 'r') as f:
    vlm_outputs = json.load(f)
with open('log_formal/qa_val_samples_video_w_desc_eval.json', 'r') as f:
    annotations = json.load(f)

# ---------- 汇总全部评论&编号 ----------
all_comments = {}
all_comments_video = {}
all_comments_task = {}
i = 0
for ann in annotations:
    for item in ann['annotations']:
        for c in item['A_hat']:
            if isinstance(c, str) and c.strip():
                #all_comments[c.strip()] = i
                all_comments[re.sub(r'^\[(?:Good Execution|Tip for Improvement)\]', '', c).strip()] = i
                all_comments_video[i] = ann['video_id']
                all_comments_task[i] = ann['scenario_name']
                i += 1
print("All comments:", len(all_comments))

ego = "ego" in file_name  # 是否为 ego clip
ego = False
# ---------- 过滤无效条目 ----------
cleaned_annotations = []
for ann in annotations:
    #if ann["scenario_name"] != args.task:
    #    continue
    ann['annotations'] = [
        item for item in ann.get('annotations', [])
        if 'video' in item and 'A_hat' in item
        and len(item.get('A_hat', [])) <= 1
    ]
    if ann['annotations']:
        cleaned_annotations.append(ann)
annotations = cleaned_annotations
print("filtered valid vidoes:", len(annotations))
valid_clip_name = set(x["video"] for ann in annotations for x in ann["annotations"])

clip_question = {(x["video"] if not ego else x["video"].split("_")[0] + ".mp4"):
                 (x["question"])
                 for ann in vlm_outputs
                 for x in ann["annotations"]
                 if x["video"] in valid_clip_name or x["video"].split("_")[0] + ".mp4" in valid_clip_name}              
print("valid question numbers:", len(clip_question))

gt_clip_question = {}
for clip in valid_clip_name:
    for ann in annotations:
        for item in ann['annotations']:
            if item.get("video") == clip and "Qe" in item:
                q_candidate = item.get("Qe", [])
    gt_clip_question[clip] = q_candidate[0]

print("valid gt question numbers:", len(gt_clip_question))

predictions = []
references = []

for clip, question in clip_question.items():
    predictions.append(question)
    references.append([gt_clip_question.get(clip, "")])

bleu_metrics = evaluate.load("bleu")
results = bleu_metrics.compute(predictions=predictions, references=references)
print(f"BLEU score: {results['bleu']:.4f}")