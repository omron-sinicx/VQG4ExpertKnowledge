
"""
Train a SentenceTransformer retriever with contrastive losses.
The training script is derived from sentence-transformers library.
"""
import json
from datasets import Dataset
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    util
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
import numpy as np
import torch
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Retriever trainng")
    parser.add_argument("--output_path", type=str, default=None, help="Save path for the trained retriever.")
    parser.add_argument("--train_file", type=str, required=True, help="json path for training data.")
    parser.add_argument("--val_file", type=str, required=True, help="json path for validation data.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Mini batch size for CachedMultipleNegativesRankingLoss.")

    args = parser.parse_args()
    return args
args = parse_args()

with open(args.train_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(args.val_file, "r", encoding="utf-8") as f:
    data_val = json.load(f)

anchors = []
postives = []
anchors_test = []
postives_test = []


# Load dataset from the EgoExoAsk annotation
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


print("Training data number:",len(train_dataset))
print("Validation data number:", len(test_dataset))

# Load base model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)


#Training config. We use a cached
loss = CachedMultipleNegativesRankingLoss(model=model, mini_batch_size=args.mini_batch_size)
train_args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=args.output_path,
    # Optional training parameters:
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_ratio=0.1,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
)
trainer = SentenceTransformerTrainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    loss=loss,
)

# Training&Save
trainer.train()
final_path = args.output_path + +"/final"
model.save_pretrained(final_path)


# Validation on test set
candidates = [item["positive"] for item in test_dataset]
queries    = [item["anchor"]   for item in test_dataset]

query_embs = model.encode(queries,  convert_to_tensor=True)
cand_embs  = model.encode(candidates, convert_to_tensor=True)

cos_scores = util.cos_sim(query_embs, cand_embs)

ranks       = []
recall_at_k = {1: 0, 5: 0, 10: 0}

for i in range(cos_scores.size(0)):
    scores     = cos_scores[i]
    sorted_idx = torch.argsort(scores, descending=True)
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