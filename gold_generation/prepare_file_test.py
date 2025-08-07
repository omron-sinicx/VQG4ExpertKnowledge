# 用户说明 seed 数据已包含 comment 附近的 descriptions，因此我们只需解析并使用它们
# 为每条注释生成模板问句、专家回答（Â）、专家提问（Q̂e）、常识提问（Q̂c）

import json
from pathlib import Path
from tqdm import tqdm
import random
from openai import OpenAI
import argparse

# 设置parse参数
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--entry", type=int, default=0, help="Entry number")
    args = parser.parse_args()
    return args
args = parse_args()

# 配置llm
openai_api_key = "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"

openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key
)

# 模拟 LLM 调用（替换为你的真实调用接口）
def llm_call(prompt: str) -> str:
    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )
    return chat_response.choices[0].message.content

# 加载已有的 qa_seed.json（每条 entry 已含 commentary + descriptions）
with open("qa_seed.json", encoding="utf-8") as f:
    seeds = json.load(f)

# QA模板
TEMPLATE_QS = {
    "good_executions": "What makes this execution good?",
    "tips_for_improvement": "How can the execution be improved?",
}

# 输出结构
training_samples = []
log_path = Path(f"log_test_{args.entry}.txt")
count = 0

# 主构造循环
for video in tqdm(seeds):
    if count == 10:
        break
    vid = video["video_id"]
    video_entry = {
        "video_id": vid,
        "annotations": []
    }
    for entry in video["annotations"]:
        ts = entry["timestamp"]
        task_type = entry["type"]
        comments = entry["commentary"]
        descriptions = entry["descriptions"]

        if task_type not in TEMPLATE_QS:
            print("skip")
            continue

        # 拼接上下文
        comment_text = " ".join(comments)
        desc_text = "\n".join(f"[{d['timestamp']:.1f}s] {d['text']}" for d in descriptions)
        q_template = TEMPLATE_QS[task_type]

        # 生成专家回答 Â
        prompt_A = f"\nExpert commentary about the video: {comment_text}\nQuestion: {q_template}\nAnswer the question based on the expert commentary.\n DO NOT use words like: based on the expert commentary. DO NOT justify your answers. DO NOT give information not mentioned in the CONTEXT INFORMATION."
        A_hat = llm_call(prompt_A)

        # 生成专家提问 Q̂e（基于 Â, C, N）
        prompt_Qe = f"You are seeing the narration of the video and the expert commentary about the video.\nNarration: {desc_text}\nExpert commentary: {comment_text}\n. Fill the question as follows:\nQ:____\nA:{A_hat}\nThe question is supposed to be raised by someone do not have expert knowledge. DO NOT justify your answers. DO NOT give information not mentioned in the CONTEXT INFORMATION."
        Qe = llm_call(prompt_Qe)

        # 生成常识提问 Q̂c（仅基于 N）
        prompt_Qc = f"You are only seeing the narration of the video: \n{desc_text}\nAsk a question based only on the narration."
        Qc = llm_call(prompt_Qc)

        # 生成四种回答
        Aee = llm_call(f"Given the video narration, answer the question referring to the exeprt commentary about the video.\nVideo narration: {desc_text}\nExpert commentary: {comment_text}\nFill in the answer as follows:\nQ:{Qe}\nA:____\nThe answer should be not TOO long. DO NOT justify your answers. DO NOT give information not mentioned in the CONTEXT INFORMATION.")
        Aec = llm_call(f"Given the video narration, answer the question.\nVideo narration: {desc_text}\nFill in the answer as follows:\nQ:{Qe}\nA:____\nThe answer should be not TOO long.")
        Ace = llm_call(f"Given the video narration, answer the question referring to the exeprt commentary about the video.\nVideo narration: {desc_text}\nExpert commentary: {comment_text}\nFill in the answer as follows:\nQ:{Qc}\nA:____\nThe answer should be not TOO long. DO NOT justify your answers. DO NOT give information not mentioned in the CONTEXT INFORMATION.")
        Acc = llm_call(f"Given the video narration, answer the question.\nVideo narration: {desc_text}\nQ:{Qc}\nA:____\nThe answer should be not TOO long.")
        # 记录日志
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"Video ID: {vid}\n")
            log_file.write(f"Timestamp: {ts}\n")
            log_file.write(f"Task Type: {task_type}\n")
            log_file.write(f"Comments: {comments}\n")
            log_file.write(f"Descriptions: {descriptions}\n")
            log_file.write(f"Template Q: {q_template}\n")
            log_file.write(f"Template A: {A_hat}\n")
            log_file.write(f"Expert Q: {Qe}\n")
            log_file.write(f"Common Q: {Qc}\n")
            log_file.write(f"Expert A on Qe: {Aee}\n")
            log_file.write(f"Common A on Qe: {Aec}\n")
            log_file.write(f"Expert A on Qc: {Ace}\n")
            log_file.write(f"Common A on Qc: {Acc}\n\n")

        video_entry["annotations"].append({
            "timestamp": ts,
            "type": task_type,
            "commentary": comments,
            "descriptions": descriptions,
            "Q_template": q_template,
            "A_hat": A_hat,
            "Qe": Qe,
            "Qc": Qc,
            "Aee": Aee,
            "Aec": Aec,
            "Ace": Ace,
            "Acc": Acc
        })
    if video_entry["annotations"]:
        training_samples.append(video_entry)
    count += 1

# 保存输出
out_path = Path(f"qa_training_samples_from_seed_entry_test_{args.entry}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(training_samples, f, indent=2, ensure_ascii=False)

