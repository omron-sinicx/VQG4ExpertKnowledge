import json
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    CrossEncoderModelCardData,
    util
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss, TripletLoss, CachedGISTEmbedLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from data import QADataset
import numpy as np
import torch
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval evaluation")
    parser.add_argument("--filename", type=str, default=None, help="filename")
    args = parser.parse_args()
    return args
args = parse_args()

with open("log_formal/qa_train_samples_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("log_formal/qa_val_samples_cleaned.json", "r", encoding="utf-8") as f:
    data_val = json.load(f)

anchors = []
postives = []
anchors_test = []
postives_test = []

for video in data:
    for ann in video["annotations"]:
        anchors.append(ann["Qe"].strip())
        postives.append(re.sub(r'^\[(?:Good Execution|Tip for Improvement)\]', '', ann["A_hat"]).strip())

for video in data_val:
    for ann in video["annotations"]:
        anchors_test.append(ann["Qe"].strip())
        postives_test.append(re.sub(r'^\[(?:Good Execution|Tip for Improvement)\]', '', ann["A_hat"]).strip())

train_dataset = Dataset.from_dict({
    "anchor": anchors,
    "positive": postives,
})

test_dataset = Dataset.from_dict({
    "anchor": anchors_test,
    "positive": postives_test,
})

#dataset = test_dataset.train_test_split(test_size=0.2, shuffle=True, seed=1)
#train_dataset = dataset["train"]
#test_dataset = dataset["test"]
print(len(train_dataset), len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
guide = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# 评价检索的 recall@k 和 rank
candidates = [item["positive"] for item in test_dataset]
queries    = [item["anchor"]   for item in test_dataset]

# 批量计算向量
query_embs = model.encode(queries,  convert_to_tensor=True)
cand_embs  = model.encode(candidates, convert_to_tensor=True)

# 相似度矩阵：shape=(len(queries), len(candidates))
cos_scores = util.cos_sim(query_embs, cand_embs)

ranks       = []
recall_at_k = {1: 0, 5: 0, 10: 0}

for i in range(cos_scores.size(0)):
    scores     = cos_scores[i]
    sorted_idx = torch.argsort(scores, descending=True)
    # 找到正确候选在排序中的位置（从1开始计）
    rank = (sorted_idx == i).nonzero(as_tuple=True)[0].item() + 1
    ranks.append(rank)
    for k in recall_at_k:
        if rank <= k:
            recall_at_k[k] += 1

num_queries = cos_scores.size(0)
avg_rank    = sum(ranks) / num_queries
for k in recall_at_k:
    recall_at_k[k] /= num_queries

print(f"Average rank: {avg_rank:.2f}")
for k in sorted(recall_at_k):
    print(f"Recall@{k}: {recall_at_k[k]:.4f}")

#loss = MultipleNegativesRankingLoss(model=model)
loss = CachedMultipleNegativesRankingLoss(model=model, mini_batch_size=128)
#loss = CachedGISTEmbedLoss(model=model, guide=guide, mini_batch_size=128, margin_strategy="absolute", margin=0.1)

train_args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="final/"+args.filename,
    # Optional training parameters:
    num_train_epochs=10,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    warmup_ratio=0.1,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    loss=loss,
)

trainer.train()
final_path = "final/"+args.filename+"/final"
model.save_pretrained(final_path)

# 评价检索的 recall@k 和 rank
candidates = [item["positive"] for item in test_dataset]
queries    = [item["anchor"]   for item in test_dataset]

# 批量计算向量
query_embs = model.encode(queries,  convert_to_tensor=True)
cand_embs  = model.encode(candidates, convert_to_tensor=True)

# 相似度矩阵：shape=(len(queries), len(candidates))
cos_scores = util.cos_sim(query_embs, cand_embs)

ranks       = []
recall_at_k = {1: 0, 5: 0, 10: 0}

for i in range(cos_scores.size(0)):
    scores     = cos_scores[i]
    sorted_idx = torch.argsort(scores, descending=True)
    # 找到正确候选在排序中的位置（从1开始计）
    rank = (sorted_idx == i).nonzero(as_tuple=True)[0].item() + 1
    ranks.append(rank)
    for k in recall_at_k:
        if rank <= k:
            recall_at_k[k] += 1

num_queries = cos_scores.size(0)
avg_rank    = sum(ranks) / num_queries
for k in recall_at_k:
    recall_at_k[k] /= num_queries

print(f"Average rank: {avg_rank:.2f}")
for k in sorted(recall_at_k):
    print(f"Recall@{k}: {recall_at_k[k]:.4f}")