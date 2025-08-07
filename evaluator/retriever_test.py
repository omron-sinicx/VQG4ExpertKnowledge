import json
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
from pathlib import Path

with open("log_formal/qa_val_samples_5.json", "r", encoding="utf-8") as f:
    data = json.load(f)

anchors = []
postives = []

for video in data:
    for ann in video["annotations"]:
        anchors.append(ann["Qe"])
        # Remove leading "[label] " prefix from the answer
        a_hat = ann["A_hat"]
        #prefix = "[label] "
        #if a_hat.startswith(prefix):
        #    a_hat = a_hat[len(prefix):]
        postives.append(a_hat)

dataset = Dataset.from_dict({
    "anchor": anchors,
    "positive": postives,
})

dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=1)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# 评价检索的 recall@k 和 rank
candidates = [item["positive"] for item in test_dataset]
queries    = [item["anchor"]   for item in test_dataset]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = SentenceTransformer("models/retriever_qwen/final", device=device)
model = SentenceTransformer("final/retriever_qwen/final", device=device)

# 批量计算向量
#query_embs = model.encode(queries, prompt="given a query, retrieve the sentences that equal to the expert insight", convert_to_tensor=True)
query_embs = model.encode(queries, convert_to_tensor=True)
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

retrieval_path = Path("retrieverlog_final/retrieval_results_qwen_ft.txt")
retrieval_path.parent.mkdir(parents=True, exist_ok=True)
# 计算定性检索结果
for i in range(len(queries)):
    with open(retrieval_path, "a", encoding="utf-8") as f:
        f.write(f"Query: {queries[i]}\n")
        f.write(f"GT: {candidates[i]}\n")
    #print(f"Top 5 candidates:")
    top_indices = torch.argsort(cos_scores[i], descending=True)[:5]
    for x, idx in enumerate(top_indices):
        with open(retrieval_path, "a", encoding="utf-8") as f:
            f.write(f"Top {x}: {candidates[idx]}, Score:{cos_scores[i][idx].item()}\n")
    with open(retrieval_path, "a", encoding="utf-8") as f:
        f.write(f"\n")