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
    args = parser.parse_args()
    return args
args = parse_args()

# ---------- 读取数据 ----------
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

all_random_r1 = []
all_random_r5 = []
all_random_r10 = []
all_random_mean_rank = []
all_random_median_rank = []
all_random_mean_entropy = []
all_random_mean_uncertainty = []

for random_seed in [1,2,3]:
    random.seed(random_seed)
    clip_question = {}
    for clip in valid_clip_name:
        for ann in annotations:
            for item in ann['annotations']:
                if item.get("video") == clip and "Qe" in item:
                    q_candidates = item.get("Qe", [])

        clip_question[clip] = random.choice(q_candidates)

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
    all_scores = []
    all_kappa = []
    recall_thresholds = [1, 5, 10]
    recall_hits = {k: 0 for k in recall_thresholds}
    positive_ranks = []
    entropy_values = []
    uncertainty = []

    # 遍历每个检索池 clip
    for clip, pool in retrieval_pools.items():
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

    median_kappa = statistics.median(all_kappa)
    print(f"Median Kappa: {median_kappa:.4f}")

    for kappa, scores in zip(all_kappa, all_scores):
        T = np.sqrt(median_kappa / np.clip(kappa, 1e-6, None))

        # 计算 softmax 概率和 entropy
        probs = F.softmax(scores, dim=0) # (M,)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        entropy_values.append(entropy)

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
            if k == 1:
                all_random_r1.append(recall)
            if k == 5:
                all_random_r5.append(recall)
            if k == 10:
                all_random_r10.append(recall)
            print(f"Recall@{k}: {recall:.4f}")
        mean_rank = sum(positive_ranks) / total
        all_random_mean_rank.append(mean_rank)
        median_rank = statistics.median(positive_ranks)
        all_random_median_rank.append(median_rank)
        print(f"Mean R: {mean_rank:.2f}")
        print(f"Median R: {median_rank}")
        print(f"Mean Entropy: {statistics.mean(entropy_values):.4f}")
        print(f"Mean Uncertainty: {statistics.mean(uncertainty):.4f}")
        all_random_mean_entropy.append(statistics.mean(entropy_values))
        all_random_mean_uncertainty.append(statistics.mean(uncertainty))

print("All random results mean:")
print(f"Mean Recall@1: {statistics.mean(all_random_r1):.4f}")
print(f"Mean Recall@5: {statistics.mean(all_random_r5):.4f}")
print(f"Mean Recall@10: {statistics.mean(all_random_r10):.4f}")
print(f"Mean Mean Rank: {statistics.mean(all_random_mean_rank):.2f}")
print(f"Mean Median Rank: {statistics.mean(all_random_median_rank):.2f}")
print(f"Mean Entropy: {statistics.mean(all_random_mean_entropy):.4f}")
print(f"Mean Uncertainty: {statistics.mean(all_random_mean_uncertainty):.4f}")

