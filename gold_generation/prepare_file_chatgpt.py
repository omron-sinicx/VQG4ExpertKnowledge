# 用户说明 seed 数据已包含 comment 附近的 descriptions，因此我们只需解析并使用它们
# 为每条注释生成模板问句、专家回答（Â）、专家提问（Q̂e）、常识提问（Q̂c）

import json
from pathlib import Path
from tqdm import tqdm
import random
from openai import OpenAI
import argparse
import re

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
    "good_executions": "What exectuion is good, or what makes this a good execution?",
    "tips_for_improvement": "What can be done to improve the execution, or what makes this a suboptimal execution?",
}
TEMPLATE_ANSWERTYPE = {
    "good_executions": "The answer is made based on the performer's good execution or why the execution is good.",
    "tips_for_improvement": "The answer is made based on the suggestions given to improve the execution or the problems about the execution.",
}
#TEMPLATE_QUESTIONTYPE = {
#    "good_executions": "DO NOT directly describe what execution is good or effective.",
#    "tips_for_improvement": "DO NOT directly mention the tips for improvement in the question.",
#}

TEMPLATE_QUESTIONTYPE = {
    "good_executions": "Your question must avoid stating that the execution was good or important, even if the commentary implies it. Instead, ask in a way that lets the answer reveal the positive aspect.",
    "tips_for_improvement": "Your question must avoid stating that the execution was bad or needed improvement, even if the commentary implies it. Instead, ask in a way that lets the answer reveal tips.",
}

# 输出结构
training_samples = []
log_path = Path(f"log_chatgpt_{args.entry}.txt")

# 主构造循环
for video in tqdm(seeds):
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

        if len(comments) < 2:
            print("skip")
            continue

        # 拼接上下文
        comment_text = " ".join(comments)
        desc_text = "\n".join(f"[{d['timestamp']:.1f}s] {d['text']}" for d in descriptions)
        q_template = TEMPLATE_QS[task_type]
        a_type = TEMPLATE_ANSWERTYPE[task_type]
        q_type = TEMPLATE_QUESTIONTYPE[task_type]

        # 生成专家回答 Â
        prompt_A = f"\nExpert commentary about the video: {comment_text}\nBased on the expert commentary, fill in the answer as follows:\nQ:{q_template}\nA:____\nDO NOT use words like: based on the expert commentary. DO NOT give information not mentioned in the CONTEXT INFORMATION.\nOutput format:\nA: ..."
        A_hat = llm_call(prompt_A)
        try:
            A_hat = re.search(r"A:\s*(.+)", A_hat).group(1).strip()
        except Exception as e:
            print(f"Error parsing A_hat: {e}")
            continue

        # 生成专家提问 Q̂e（基于 Â, C, N）
        #prompt_Qe = f"You are seeing the narration of the video and the expert commentary about the video.\nNarration: {desc_text}\nExpert commentary: {comment_text}\n. Fill in the question as follows:\nQ:____\nA:{A_hat}\n{a_type}\nThe question is supposed to be raised by someone do not have expert knowledge.\nDO NOT give information not mentioned in the CONTEXT INFORMATION.\n{q_type}\nOutput format:\n[question] ..."
        #prompt_Qe = f"You are seeing a task video.\nNarration of the video:\n{desc_text}\nExpert commentary of the video: {comment_text}\n.Fill in the question as follows:\nQ:____\nA:{A_hat}\n{a_type}\nThe question is supposed to be raised by someone do not have expert knowledge. The question should be high-quality.\nAVOID vague phrases like “the video,” “this movement,” or “the performance.” DO NOT ask for general evaluations, summaries, or opinions. Each question should focus on a concrete moment, gesture, or object in the scene.\nDO NOT use words like: based on the expert commentary. DO NOT give information not mentioned in the CONTEXT INFORMATION.\n{q_type}\nOutput format:\nQ: ..."
        prompt_Qe = f"""You are an AI assistant tasked with generating insightful questions. You will be observing a task video, presented as a "Narration" (describing the surface-level actions) and an "Expert Commentary" (providing deeper insights, reasons, or techniques which are considered the implicit knowledge in this context).

Narration of the video:
{desc_text}

Expert commentary of the video:
{comment_text}

**Context for the question to be generated**:
{a_type}

Your goal is to formulate a question that a curious novice, after seeing the "Narration", might ask to understand the deeper insights revealed in the "Expert Commentary", keeping in mind the specific context provided above. The question should bridge the gap between what is simply observed (in the Narration) and why it's significant or done in a particular way (as explained in the Expert Commentary). The question should ultimately guide the learner towards an understanding similar to what an expert might articulate (conceptually related to the answer, but the question must not presuppose answer).

Fill in the question as follows:
Q:____
A:{A_hat}

Here are the guidelines for crafting the question:

1.  **Perspective**: The question should be from the perspective of someone who does *not* have expert knowledge but is eager to learn the underlying expert reasoning.
2.  **High-Quality & Depth**: A high-quality question should:
    * Focus on a concrete moment, gesture, or object described in the "Narration".
    * Go beyond what is obvious from the "Narration" alone. It should probe into the *reasons*, *intentions*, *subtle techniques*, or *critical judgments* highlighted or implied by the "Expert Commentary".
    * Help a learner understand *why* the expert's actions or observations are important or effective (if the commentary refers to good execution) OR *what specific adjustments could lead to better outcomes* (if the commentary refers to tips for improvement).
    * Essentially, the answer to the question should highlight the value or insight provided by the "Expert Commentary".
3.  **Adherence to Question Type Constraint**:
    * **Crucially, {q_type}**
4.  **Clarity and Specificity**:
    * AVOID vague phrases like “the video,” “this movement,” or “the performance.”
    * DO NOT ask for general evaluations, summaries, or opinions.
5.  **Information Grounding**:
    * DO NOT give information or ask about information not mentioned in the provided "Narration" or "Expert Commentary".
6.  **Implicit Knowledge Focus**: The question should aim to elicit or clarify the kind of knowledge that distinguishes an expert from a novice, as revealed by the "Expert Commentary".

Output format:
Q: ..."""
        Qe = llm_call(prompt_Qe)
        try:
            Qe = re.search(r"Q:\s*(.+)", Qe).group(1).strip()
        except Exception as e:
            print(f"Error parsing Qe: {e}")
            continue

        # 生成常识提问 Q̂c（仅基于 N）
        #prompt_Qc = f"You are only seeing the narration of the video: \n{desc_text}\nAsk a question based only on the narration."
        prompt_Qc = f"You are seeing a task video.\nNarration of the video:\n{desc_text}\nAsk a question based only on the narration."
        Qc = llm_call(prompt_Qc)

        # 生成四种回答
        Aee = llm_call(f"Given the video narration, answer the question referring to the exeprt commentary about the video.\nVideo narration: {desc_text}\nExpert commentary: {comment_text}\nFill in the answer as follows:\nQ:{Qe}\nA:____\nDO NOT give information not mentioned in the CONTEXT INFORMATION. The answer should be not TOO long. DO NOT use words like: as mentioned in the expert commentary. DO NOT repeat words in the question.\nOutput format:\nA: ...")
        try:
            Aee = re.search(r"A:\s*(.+)", Aee).group(1).strip()
        except Exception as e:
            print(f"Error parsing Aee: {e}")
            continue
        Aec = llm_call(f"Given the video narration, answer the question.\nVideo narration: {desc_text}\nFill in the answer as follows:\nQ:{Qe}\nA:____\nThe answer should be not TOO long. DO NOT repeat words in the question.\nOutput format:\nA: ...")
        try:
            Aec = re.search(r"A:\s*(.+)", Aec).group(1).strip()
        except Exception as e:
            print(f"Error parsing Aec: {e}")
            continue
        Ace = llm_call(f"Given the video narration, answer the question referring to the exeprt commentary about the video.\nVideo narration: {desc_text}\nExpert commentary: {comment_text}\nFill in the answer as follows:\nQ:{Qc}\nA:____\nDO NOT give information not mentioned in the CONTEXT INFORMATION. The answer should be not TOO long. DO NOT use words like: as mentioned in the expert commentary.\nOutput format:\nA: ...")
        try:
            Ace = re.search(r"A:\s*(.+)", Ace).group(1).strip()
        except Exception as e:
            print(f"Error parsing Ace: {e}")
            continue
        Acc = llm_call(f"Given the video narration, answer the question.\nVideo narration: {desc_text}\nFill in the answer as follows:\nQ:{Qc}\nA:____\nThe answer should be not TOO long.\nOutput format:\nA: ...")
        try:
            Acc = re.search(r"A:\s*(.+)", Acc).group(1).strip()
        except Exception as e:
            print(f"Error parsing Acc: {e}")
            continue
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

# 保存输出
out_path = Path(f"qa_training_samples_from_seed_chatgpt_entry_{args.entry}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(training_samples, f, indent=2, ensure_ascii=False)

