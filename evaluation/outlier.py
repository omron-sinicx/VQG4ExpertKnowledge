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
import os

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

# ---------- 过滤无效条目 ----------
cleaned_annotations = []
for ann in annotations:
    #if ann["scenario_name"] != args.task:
    #    continue
    ann['annotations'] = [
        item for item in ann.get('annotations', [])
        if 'video' in item and 'A_hat' in item
        #and len(item.get('A_hat', [])) <= 1
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

# ---------- clip -> 正样本 & 记录 clip 对应的 video_id和scenario ----------
clip2positives = defaultdict(list)
clip2scenario = {}
clip2video = {}
for ann in annotations:
    for item in ann['annotations']:
        vid = item['video']
        clip2positives[vid].extend(
            [re.sub(r'^\[(?:Good Execution|Tip for Improvement)\]', '', c).strip() for c in item['A_hat'] if isinstance(c, str)]
        )
        clip2video[vid] = ann["video_id"]
        clip2scenario[vid] = ann['scenario_name']

print("clip2positives size:", len(clip2positives))

# ---------- 编码所有 comment和query ----------
model = SentenceTransformer('final/retriever_st_2/final')
#model = SentenceTransformer('final/st_GIST/final')
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
comment_embs = model.encode(
    list(all_comments.keys()),
    convert_to_tensor=True
)
# 标准化emb
comment_embs = comment_embs.cpu()  # 确保在 CPU 上

query_embs = model.encode(
    list(clip_question.values()),
    convert_to_tensor=True
) # (N, dim)
query_embs = query_embs.cpu()  # 确保在 CPU 上

query_embs_dict = {clip: emb for clip, emb in zip(clip_question.keys(), query_embs)}

# ---------- 为每个 clip 构造检索池 ----------
random.seed(42)

retrieval_pools = {}
for clip, pos_comments in clip2positives.items():
    question = clip_question.get(clip, "")
    # 正样本 id 列表
    pos_ids = [all_comments[c] for c in pos_comments]
    n = len(pos_ids)
    # 决定负采样数：n*6<15 则 15−n，否则 5*n
    #neg_target = n*6
    neg_target = 50 - n  # 固定为 30 条负样本
    neg_ids = []

    # 1) 同 video 负样本
    v = clip2video[clip]
    same_vid = [idx for idx, vid in all_comments_video.items()
                if vid == v and idx not in pos_ids]
    random.shuffle(same_vid)
    take = min(len(same_vid), neg_target)
    neg_ids.extend(same_vid[:take])
    rem = neg_target - take

    # 2) 同 scenario 负样本
    if rem > 0:
        task = clip2scenario[clip]
        same_task = [idx for idx, t in all_comments_task.items()
                     if t == task and all_comments_video[idx] != v and idx not in pos_ids]
        random.shuffle(same_task)
        take2 = min(len(same_task), rem)
        neg_ids.extend(same_task[:take2])
        rem -= take2

    # 3) 随机补齐
    if rem > 0:
        others = [idx for idx in all_comments_video.keys()
                  if idx not in pos_ids and idx not in neg_ids]
        random.shuffle(others)
        neg_ids.extend(others[:rem])

    all_ids = pos_ids + neg_ids
    retrieval_pools[clip] = {
        "question": question,
        "positive_ids": pos_ids,
        "negative_ids": neg_ids,
        "all_ids": all_ids
    }


# 从 all_comments 构建 id -> comment 文本的映射
#idx2comment = {idx: text for text, idx in all_comments.items()}
## 随机选 5 个 clip
#sample_clips = random.sample(list(retrieval_pools.keys()), 5)
## 打印每个 clip 的检索池文本
#for clip in sample_clips:
#    pool = retrieval_pools[clip]
#    all_ids = pool["all_ids"]
#    print(f"--- Clip: {clip} 检索池 ({len(all_ids)} 条) ---")
#    for cid in all_ids:
#        print(idx2comment[cid])
#    print()

import torch.nn.functional as F

# 定义不同的 k 值
recall_thresholds = [1, 5, 10]
recall_hits = {k: 0 for k in recall_thresholds}
positive_ranks = []
entropy_values = []
norm_entropy_values = []
uncertainty = []

# 遍历每个检索池 clip
for clip, pool in retrieval_pools.items():
    # 获取当前 clip 的查询 embedding
    question_emb = query_embs_dict.get(clip)
    question_emb = question_emb.unsqueeze(0)  # (1, dim)
    # 获取检索候选的 comment id 列表和对应 embedding
    candidate_ids = pool["all_ids"]
    candidate_embs = comment_embs[candidate_ids] # (M, dim)

    # 对 candidate 和 query 做计算余弦相似度
    scores = cosine_similarity(
        question_emb,
        candidate_embs
    )  # (1, M)
    scores = torch.tensor(scores).squeeze(0)  # 转为 Tensor (M,)

    # 对候选条目按相似度降序排序，获取排序后的索引 (相对于 candidate_ids 列表)
    sorted_indices = torch.argsort(scores, descending=True)
    
    # 找出所有正样本在候选列表中的位置（排序中最低排名即为最佳匹配）
    pos_set = set(pool["positive_ids"])
    best_rank = None  # 1 基数
    for rank, idx in enumerate(sorted_indices.tolist(), start=1):
        if candidate_ids[idx] in pos_set:
            best_rank = rank
            break
    positive_ranks.append(best_rank)

    # 更新各 recall@k
    for k in recall_thresholds:
        if best_rank <= k:
            recall_hits[k] += 1

    # 计算 softmax 概率和 entropy
    probs = F.softmax(scores, dim=0) # (M,)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    norm_entropy = entropy / np.log(len(candidate_ids))
    entropy_values.append(entropy)
    norm_entropy_values.append(norm_entropy)

    # 计算不确定性
    m = torch.nn.ReLU()
    s2 = m(scores)
    evidence = torch.exp(s2) # (M,)
    s = torch.sum(evidence + 1)
    unc = len(candidate_ids) / s.item()  # 归一化不确定性
    uncertainty.append(unc)



total = len(positive_ranks)
if total == 0:
    print("invalid")
else:
    print("Results:")
    for k in recall_thresholds:
        recall = recall_hits[k] / total
        print(f"Recall@{k}: {recall:.4f}")
    mean_rank = sum(positive_ranks) / total
    median_rank = statistics.median(positive_ranks)
    print(f"Mean R: {mean_rank:.2f}")
    print(f"Median R: {median_rank}")
    print(f"Mean Entropy: {statistics.mean(entropy_values):.4f}")
    print(f"Mean Norm Entropy: {statistics.mean(norm_entropy_values):.4f}")
    print(f"Mean Uncertainty: {statistics.mean(uncertainty):.4f}")

# 构造 id -> comment 文本的映射
id2comment = {}
for text, idx in all_comments.items():
    id2comment[idx] = text
# 收集每个检索池的详细信息，包括正样本排名和 top-10 检索结果
detailed_results = []
all_sim = []
for clip, pool in retrieval_pools.items():
    question_text = pool["question"]
    candidate_ids = pool["all_ids"]
    # 获取查询 embedding
    question_emb = query_embs_dict.get(clip)
    if question_emb is None:
        continue
    question_emb = question_emb.unsqueeze(0)
    candidate_embs = comment_embs[candidate_ids]
    scores = cosine_similarity(question_emb, candidate_embs)
    scores = torch.tensor(scores).squeeze(0)
    sorted_indices = torch.argsort(scores, descending=True)

    probs = F.softmax(scores, dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    norm_entropy = entropy / np.log(len(candidate_ids))

    # 计算不确定性
    m = torch.nn.ReLU()
    s2 = m(scores)
    evidence = torch.exp(s2) # (M,)
    s = torch.sum(evidence + 1)
    unc = len(candidate_ids) / s.item()  # 归一化不确定性
    
    # 找到正样本的最优排名
    pos_set = set(pool["positive_ids"])
    best_rank = None
    for rank, idx in enumerate(sorted_indices.tolist(), start=1):
        if candidate_ids[idx] in pos_set:
            best_rank = rank
            break
    if best_rank is None:
        continue
    top10_results = []
    # 获取 top-5 检索结果文本
    top10_comments = []
    for idx in sorted_indices.tolist()[:5]:
        comment_id = candidate_ids[idx]
        comment_text = id2comment.get(comment_id, "<missing>")
        sim_score = scores[idx].item()
        top10_results.append({"text": comment_text, "similarity": sim_score})

    top1_sim = top10_results[0]["similarity"]
    top1_question = top10_results[0]["text"]
    all_sim.append({"text": top1_question, "similarity":top1_sim})

    
    # 添加正确答案文本
    correct_answers = [id2comment.get(pid, "<missing>") for pid in pool["positive_ids"]]
    
    detailed_results.append({
        "clip": clip,
        "question": question_text,
        "best_rank": best_rank,
        "top10": top10_results,
        "correct_answers": correct_answers,
        "entropy": entropy,
        "norm_entropy": norm_entropy,
        "uncertainty": unc
    })

# 将all_sim中的相似度plot在1D数轴上
import matplotlib.pyplot as plt

# 提取相似度分数
scores = [item["similarity"] for item in all_sim]

# 在一条数轴上用散点展示

plt.figure(figsize=(10, 2))
plt.scatter(scores, [0] * len(scores), alpha=0.6)
plt.yticks([])                              # 隐藏 y 轴刻度
plt.xlim(0.0, 1.0)                          # 设置 x 轴范围为 [0.0, 1.0]
plt.xticks(np.arange(0.0, 1.01, 0.1))       # x 轴刻度以 0.1 为单位
plt.xlabel("Similarity Score")
plt.title("1D Distribution of Similarity Scores")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.savefig("similarity/similarity_" + args.filename + ".png")