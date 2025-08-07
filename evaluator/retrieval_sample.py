import json
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
from pathlib import Path

query = "Why does the player take a few steps back before jumping to shoot the basketball?"

with open("log_formal/qa_val_samples_5.json", "r", encoding="utf-8") as f:
    data = json.load(f)

anchors = []
postives = []

for video in data:
    for ann in video["annotations"]:
        anchors.append(ann["Qe"])
        postives.append(ann["A_hat"])

dataset = Dataset.from_dict({
    "anchor": anchors,
    "positive": postives,
})

# 评价检索的 recall@k 和 rank
candidates = [item["positive"] for item in dataset]
queries    = [item["anchor"]   for item in dataset]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = SentenceTransformer("models/retriever_qwen/final", device=device)
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)

query_embs = model.encode(query, convert_to_tensor=True)
cand_embs  = model.encode(candidates, convert_to_tensor=True)

# 相似度矩阵：shape=(len(queries), len(candidates))
cos_scores = util.cos_sim(query_embs, cand_embs)

# 计算定性检索结果
print(f"Query: {query}\n")
print(f"Top 5 candidates:")
top_indices = torch.argsort(cos_scores[0], descending=True)[:5]
for x, idx in enumerate(top_indices):
    print(f"Top {x+1}: {candidates[idx]}, Score:{cos_scores[0][idx].item()}\n")