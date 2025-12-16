"""
Retrieval evaluation script.
"""

import argparse
import json
import os
import random
import re
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


TAG_PREFIX_RE = re.compile(r"^\[(?:Good Execution|Tip for Improvement)\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval evaluation for clip-comment retrieval.")

    # Input / output
    parser.add_argument("--vqg_json", type=str, required=True,
                        help="Path to your VQG output json.")
    parser.add_argument("--ann_json", type=str, required=True,
                        help="Path to the EgoExoAsk annotations json.")
    parser.add_argument("--output_txt", type=str, default="evaluation_results.txt",
                        help="Output text file to write detailed results.")

    # Model
    parser.add_argument("--retriever_model", type=str, required=True,
                        help="SentenceTransformer model name or local path.")

    # Behavior
    parser.add_argument("--seed", type=int, default=42, help="Random seed for negative sampling. Set 42 for reproducibility of our paper.")
    parser.add_argument("--pool_size", type=int, default=50,
                        help="Total retrieval pool size per clip (positives + negatives).")
    parser.add_argument("--recall_ks", type=int, nargs="+", default=[1, 5, 10],
                        help="K values for Recall@K.")
    parser.add_argument("--topk_dump", type=int, default=5,
                        help="How many top retrieved comments to dump per clip in the qualitative output.")
    parser.add_argument("--good_dump_n", type=int, default=50,
                        help="How many best-rank clips to dump.")
    parser.add_argument("--worst_dump_n", type=int, default=10,
                        help="How many worst-rank clips to dump.")

    return parser.parse_args()


def strip_tag_prefix(text: str) -> str:
    """Remove leading '[Good Execution]' / '[Tip for Improvement]' tags and whitespace."""
    return TAG_PREFIX_RE.sub("", text).strip()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_valid_annotations(annotations: List[dict]) -> List[dict]:
    """
    Keep only entries whose items contain required keys ('video', 'A_hat').
    """
    cleaned = []
    for ann in annotations:
        ann["annotations"] = [
            item for item in ann.get("annotations", [])
            if ("video" in item) and ("A_hat" in item)
        ]
        if ann["annotations"]:
            cleaned.append(ann)
    return cleaned


def build_comment_tables(
    annotations: List[dict],
) -> Tuple[Dict[str, int], Dict[int, str], Dict[int, str], Dict[int, str]]:
    """
    Build:
      - comment_text_to_id: {comment_text: global_id}
      - id_to_video_id: {global_id: video_id}
      - id_to_scenario: {global_id: scenario_name}
      - id_to_comment_text: {global_id: comment_text}
    """
    comment_text_to_id: Dict[str, int] = {}
    id_to_video_id: Dict[int, str] = {}
    id_to_scenario: Dict[int, str] = {}
    id_to_comment_text: Dict[int, str] = {}

    i = 0
    for ann in annotations:
        for item in ann["annotations"]:
            for c in item.get("A_hat", []):
                if not isinstance(c, str):
                    continue
                c = c.strip()
                if not c:
                    continue
                c = strip_tag_prefix(c)
                # Keep the first ID assigned if duplicates occur
                if c in comment_text_to_id:
                    continue
                comment_text_to_id[c] = i
                id_to_video_id[i] = ann.get("video_id", "")
                id_to_scenario[i] = ann.get("scenario_name", "")
                id_to_comment_text[i] = c
                i += 1

    return comment_text_to_id, id_to_video_id, id_to_scenario, id_to_comment_text


def build_clip_tables(
    annotations: List[dict],
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], set]:
    """
    Build:
      - clip_to_positive_comments: {clip_name: [comment_text, ...]}
      - clip_to_scenario: {clip_name: scenario_name}
      - clip_to_video_id: {clip_name: video_id}
      - valid_clip_names: set of all valid clip names in annotations
    """
    clip_to_positive_comments: Dict[str, List[str]] = defaultdict(list)
    clip_to_scenario: Dict[str, str] = {}
    clip_to_video_id: Dict[str, str] = {}

    valid_clip_names = set()
    for ann in annotations:
        for item in ann["annotations"]:
            vid = item["video"]
            valid_clip_names.add(vid)
            clip_to_video_id[vid] = ann.get("video_id", "")
            clip_to_scenario[vid] = ann.get("scenario_name", "")
            positives = [
                strip_tag_prefix(c).strip()
                for c in item.get("A_hat", [])
                if isinstance(c, str) and c.strip()
            ]
            clip_to_positive_comments[vid].extend(positives)

    return clip_to_positive_comments, clip_to_scenario, clip_to_video_id, valid_clip_names


def load_vqg_questions(
    vqg_outputs: List[dict],
    valid_clip_names: set,
) -> Dict[str, str]:
    """
    Extract {clip_name: question_text} from vqg outputs, filtering to valid clips.
    """
    clip_to_question: Dict[str, str] = {}

    for ann in vqg_outputs:
        for x in ann.get("annotations", []):
            if "video" not in x or "question" not in x:
                continue
            v = x["video"]
            q = x["question"]
            if not isinstance(q, str):
                continue

            # Map: 'abc_def...' -> 'abc.mp4'
            base = v.split("_")[0] + ".mp4"
            if (v in valid_clip_names) or (base in valid_clip_names):
                clip_to_question[base if base in valid_clip_names else v] = q

    return clip_to_question

def build_retrieval_pools(
    clip_to_positive_comments: Dict[str, List[str]],
    clip_to_question: Dict[str, str],
    comment_text_to_id: Dict[str, int],
    id_to_video_id: Dict[int, str],
    id_to_scenario: Dict[int, str],
    clip_to_video_id: Dict[str, str],
    clip_to_scenario: Dict[str, str],
    pool_size: int,
    seed: int,
) -> Dict[str, dict]:
    """
    Build retrieval pools per clip:
      positives + negatives = pool_size

    Negatives sampling priority:
      1) same video_id
      2) same scenario, different video_id
      3) anywhere else
    """
    random.seed(seed)

    retrieval_pools: Dict[str, dict] = {}
    all_comment_ids = list(id_to_video_id.keys())

    for clip, pos_comments in clip_to_positive_comments.items():
        if clip not in clip_to_question:
            continue

        # Map positive comment texts to IDs (drop those not found due to filtering/dedup)
        pos_ids = []
        for c in pos_comments:
            c = c.strip()
            if c in comment_text_to_id:
                pos_ids.append(comment_text_to_id[c])
        pos_ids = list(dict.fromkeys(pos_ids))  # unique preserve order

        n_pos = len(pos_ids)
        if n_pos == 0:
            continue

        neg_target = max(pool_size - n_pos, 0)
        neg_ids: List[int] = []

        v = clip_to_video_id.get(clip, "")
        task = clip_to_scenario.get(clip, "")

        # 1) same video_id negatives
        same_vid = [idx for idx, vid in id_to_video_id.items()
                    if (vid == v) and (idx not in pos_ids)]
        random.shuffle(same_vid)
        take = min(len(same_vid), neg_target)
        neg_ids.extend(same_vid[:take])
        rem = neg_target - take

        # 2) same scenario (task), different video_id negatives
        if rem > 0:
            same_task = [idx for idx, t in id_to_scenario.items()
                         if (t == task) and (id_to_video_id[idx] != v) and (idx not in pos_ids)]
            random.shuffle(same_task)
            take2 = min(len(same_task), rem)
            neg_ids.extend(same_task[:take2])
            rem -= take2

        # 3) fill from anywhere
        if rem > 0:
            others = [idx for idx in all_comment_ids
                      if (idx not in pos_ids) and (idx not in neg_ids)]
            random.shuffle(others)
            neg_ids.extend(others[:rem])

        all_ids = pos_ids + neg_ids

        retrieval_pools[clip] = {
            "question": clip_to_question.get(clip, ""),
            "positive_ids": pos_ids,
            "negative_ids": neg_ids,
            "all_ids": all_ids,
        }

    return retrieval_pools


def compute_metrics_for_pools(
    retrieval_pools: Dict[str, dict],
    comment_embs: torch.Tensor,
    query_embs_dict: Dict[str, torch.Tensor],
    recall_ks: List[int],
) -> Dict[str, Any]:
    """
    Compute Recall@K and best positive rank
    """
    recall_hits = {k: 0 for k in recall_ks}
    positive_ranks: List[int] = []

    for clip, pool in retrieval_pools.items():
        q_emb = query_embs_dict.get(clip)
        if q_emb is None:
            continue
        q_emb = q_emb.unsqueeze(0)  # (1, D)

        candidate_ids = pool["all_ids"]
        if len(candidate_ids) == 0:
            continue

        cand_embs = comment_embs[candidate_ids]  # (M, D)

        scores = cosine_similarity(q_emb, cand_embs)  # (1, M)
        scores = torch.tensor(scores).squeeze(0)      # (M,)

        sorted_indices = torch.argsort(scores, descending=True).tolist()

        pos_set = set(pool["positive_ids"])
        best_rank = None
        for rank, idx in enumerate(sorted_indices, start=1):
            if candidate_ids[idx] in pos_set:
                best_rank = rank
                break
        if best_rank is None:
            continue

        positive_ranks.append(best_rank)

        for k in recall_ks:
            if best_rank <= k:
                recall_hits[k] += 1

    total = len(positive_ranks)
    if total == 0:
        return {"total": 0}

    results = {
        "total": total,
        "recall": {k: recall_hits[k] / total for k in recall_ks},
        "mean_rank": float(sum(positive_ranks) / total),
        "median_rank": int(statistics.median(positive_ranks)),
        # Keep for later dumps if needed
        "positive_ranks": positive_ranks,
    }
    return results


def build_detailed_results(
    retrieval_pools: Dict[str, dict],
    comment_embs: torch.Tensor,
    query_embs_dict: Dict[str, torch.Tensor],
    id_to_comment_text: Dict[int, str],
    topk_dump: int,
) -> List[dict]:
    """
    Build detailed per-clip results for qualitative inspection.
    """
    detailed = []

    for clip, pool in retrieval_pools.items():
        q_text = pool["question"]
        q_emb = query_embs_dict.get(clip)
        if q_emb is None:
            continue
        q_emb = q_emb.unsqueeze(0)

        candidate_ids = pool["all_ids"]
        if len(candidate_ids) == 0:
            continue
        cand_embs = comment_embs[candidate_ids]

        scores = cosine_similarity(q_emb, cand_embs)
        scores = torch.tensor(scores).squeeze(0)
        sorted_indices = torch.argsort(scores, descending=True).tolist()

        pos_set = set(pool["positive_ids"])
        best_rank = None
        for rank, idx in enumerate(sorted_indices, start=1):
            if candidate_ids[idx] in pos_set:
                best_rank = rank
                break
        if best_rank is None:
            continue

        topk = []
        for idx in sorted_indices[:topk_dump]:
            cid = candidate_ids[idx]
            topk.append({
                "text": id_to_comment_text.get(cid, "<missing>"),
                "similarity": float(scores[idx].item()),
            })

        correct_answers = [id_to_comment_text.get(pid, "<missing>") for pid in pool["positive_ids"]]

        detailed.append({
            "clip": clip,
            "question": q_text,
            "best_rank": int(best_rank),
            "topk": topk,
            "correct_answers": correct_answers,
        })

    return detailed


def qualitative_results(
    path: str,
    metrics: Dict[str, Any],
    detailed_results: List[dict],
    recall_ks: List[int],
    good_dump_n: int,
    worst_dump_n: int,
):
    if os.path.exists(path):
        os.remove(path)

    # Sort detailed outputs
    sorted_best = sorted(detailed_results, key=lambda x: x["best_rank"])
    sorted_worst = sorted(detailed_results, key=lambda x: x["best_rank"], reverse=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("Results:\n")
        for k in recall_ks:
            f.write(f"Recall@{k}: {metrics['recall'][k]:.4f}\n")
        f.write(f"Mean Rank: {metrics['mean_rank']:.2f}\n")
        f.write(f"Median Rank: {metrics['median_rank']}\n")

        f.write("Good Samples (best rank):\n")
        for entry in sorted_best[:good_dump_n]:
            f.write(f"\nClip: {entry['clip']}\n")
            f.write(f"Q: {entry['question']}\n")
            f.write(f"Best Rank: {entry['best_rank']}\n")
            f.write(f"GT: {', '.join(entry['correct_answers'])}\n")
            f.write("Top Results:\n")
            for i, item in enumerate(entry["topk"], start=1):
                f.write(f"  {i}: {item['similarity']:.4f} | {item['text']}\n")

        f.write("\nWorst Samples (worst rank):\n")
        for entry in sorted_worst[:worst_dump_n]:
            f.write(f"\nClip: {entry['clip']}\n")
            f.write(f"Q: {entry['question']}\n")
            f.write(f"Best Rank: {entry['best_rank']}\n")
            f.write(f"GT: {', '.join(entry['correct_answers'])}\n")
            f.write("Top Results:\n")
            for i, item in enumerate(entry["topk"], start=1):
                f.write(f"  {i}: {item['similarity']:.4f} | {item['text']}\n")


def main():
    args = parse_args()

    # 1) Load data
    vqg_outputs = load_json(args.vqg_json)
    annotations = load_json(args.ann_json)

    # 2) Filter invalid annotation items
    annotations = filter_valid_annotations(annotations)
    valid_clip_names = set(x["video"] for ann in annotations for x in ann["annotations"])

    # 3) Build tables
    (comment_text_to_id,
     id_to_video_id,
     id_to_scenario,
     id_to_comment_text) = build_comment_tables(annotations)

    (clip_to_positive_comments,
     clip_to_scenario,
     clip_to_video_id,
     _) = build_clip_tables(annotations)

    clip_to_question = load_vqg_questions(vqg_outputs, valid_clip_names)

    print(f"Valid videos: {len(annotations)}")
    print(f"All unique comments: {len(comment_text_to_id)}")
    print(f"Valid questions: {len(clip_to_question)}")

    # 4) Encode texts
    model = SentenceTransformer(args.retriever_model)

    all_comment_texts = list(comment_text_to_id.keys())
    comment_embs = model.encode(all_comment_texts, convert_to_tensor=True)
    comment_embs = comment_embs.cpu()

    all_questions = list(clip_to_question.values())
    query_embs = model.encode(all_questions, convert_to_tensor=True)
    query_embs = query_embs.cpu()
    query_embs_dict = {clip: emb for clip, emb in zip(clip_to_question.keys(), query_embs)}

    # 5) Build retrieval pools
    retrieval_pools = build_retrieval_pools(
        clip_to_positive_comments=clip_to_positive_comments,
        clip_to_question=clip_to_question,
        comment_text_to_id=comment_text_to_id,
        id_to_video_id=id_to_video_id,
        id_to_scenario=id_to_scenario,
        clip_to_video_id=clip_to_video_id,
        clip_to_scenario=clip_to_scenario,
        pool_size=args.pool_size,
        seed=args.seed,
    )
    print(f"Retrieval pools: {len(retrieval_pools)}")

    # 6) Compute metrics
    metrics = compute_metrics_for_pools(
        retrieval_pools=retrieval_pools,
        comment_embs=comment_embs,
        query_embs_dict=query_embs_dict,
        recall_ks=args.recall_ks,
    )

    if metrics.get("total", 0) == 0:
        print("No valid samples to evaluate.")
        return

    print("Results:")
    for k in args.recall_ks:
        print(f"Recall@{k}: {metrics['recall'][k]:.4f}")
    print(f"Mean Rank: {metrics['mean_rank']:.2f}")
    print(f"Median Rank: {metrics['median_rank']}")

    # 7) Qualitative results dump
    detailed_results = build_detailed_results(
        retrieval_pools=retrieval_pools,
        comment_embs=comment_embs,
        query_embs_dict=query_embs_dict,
        id_to_comment_text=id_to_comment_text,
        topk_dump=args.topk_dump,
    )

    qualitative_results(
        path=args.output_txt,
        metrics=metrics,
        detailed_results=detailed_results,
        recall_ks=args.recall_ks,
        good_dump_n=args.good_dump_n,
        worst_dump_n=args.worst_dump_n,
    )
    print(f"Written to: {args.output_txt}")


if __name__ == "__main__":
    main()