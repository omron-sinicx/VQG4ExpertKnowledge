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
from scipy.stats import vonmises, norm, vonmises_fisher
import matplotlib.pyplot as plt

import torch.nn.functional as F

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval evaluation")
    parser.add_argument("--task", type=str, default="all", help="Scenario name")
    parser.add_argument("--filename", type=str, default=None, help="filename")
    args = parser.parse_args()
    return args
args = parse_args()

# ---------- 读取数据 ----------
if args.filename != "gold":
    file_name = "log_formal/"+args.filename+".json"
    with open(file_name, 'r') as f:
        vlm_outputs = json.load(f)
    ego = "ego" in file_name  # 是否为 ego clip
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

if args.filename == "gold":
    random.seed(42)
    clip_question = {}
    for clip in valid_clip_name:
        for ann in annotations:
            for item in ann['annotations']:
                if item.get("video") == clip and "Qe" in item:
                    q_candidates = item.get("Qe", [])
        clip_question[clip] = random.choice(q_candidates)

else:
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

all_scores = []
uncertainty = []
covariance = []
all_kappa = []
all_kappa_hat = []
for i, (clip, pool) in enumerate(retrieval_pools.items()):
    # 获取当前 clip 的查询 embedding
    question_emb = query_embs_dict.get(clip)
    question_emb = question_emb.unsqueeze(0)  # (1, dim)
    # 获取检索候选的 comment id 列表和对应 embedding
    candidate_ids = pool["all_ids"]
    candidate_embs = comment_embs[candidate_ids] # (M, dim)
    r = candidate_embs.mean(dim=0)
    R = torch.linalg.norm(r)
    dim = candidate_embs.shape[1]
    kappa = R * (dim - R ** 2) / (1 - R ** 2)
    all_kappa.append(kappa.item())

    # 对 candidate 和 query 做计算余弦相似度
    scores = cosine_similarity(
        question_emb,
        candidate_embs
    )  # (1, M)

    scores = torch.tensor(scores).squeeze(0)  # 转为 Tensor (M,)
    all_scores.append(scores)

median_kappa = statistics.median(all_kappa)
print(f"Median Kappa: {median_kappa:.4f}")

all_entropy = []
positive_ranks = []

for kappa, scores in zip(all_kappa, all_scores):
    #T = np.sqrt(median_kappa / np.clip(kappa, 1e-6, None))
    T = median_kappa / np.clip(kappa, 1e-6, None)

    probs = F.softmax(scores / T, dim=0)  # 使用 softmax 计算概率
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    all_entropy.append(entropy)


    m = torch.nn.ReLU()
    s2 = m(scores / T)
    evidence = torch.exp(s2)
    s = torch.sum(evidence+1)
    unc = len(candidate_ids) / s.item()
    uncertainty.append(unc)

print(f"Mean Normalized Entropy: {statistics.mean(all_entropy):.4f}")
print(f"Uncertainty: {statistics.mean(uncertainty):.4f}")





#for i, s in enumerate(all_scores):
#    plt.subplot(2, 2, i + 1)
#    plt.hist(s, bins=20, density=True)
#    kappa, loc, scale = vonmises.fit(s)
#    mean, std = norm.fit(s)
#    x = np.linspace(-1, 1, 1000)
#    y = vonmises.pdf(x, kappa, loc, scale=scale)
#    y = norm.pdf(x, loc=mean, scale=std)
#    print(kappa, loc)
#    plt.plot(x, y, color='red')

#plt.savefig('cosine_similarity_distribution_gaussian_pools.png')