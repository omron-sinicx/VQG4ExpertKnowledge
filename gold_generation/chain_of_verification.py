import json
from pathlib import Path
from tqdm import tqdm
import random
from openai import OpenAI
import argparse
import re

TEMPLATE_QUESTIONTYPE = {
    "good_executions": f"""The aim of the question is to uncover the notifiable actions or techniques.""",
    "tips_for_improvement": f"""The aim of the question is to uncover the actions that are not optimal.""",
}

#checker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# 设置parse参数
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--entry", type=str, default=0, help="Entry number")
    args = parser.parse_args()
    return args
args = parse_args()

# 配置llm
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 模拟 LLM 调用（替换为你的真实调用接口）
def llm_call(prompt: str) -> str:
    chat_response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20, 
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return chat_response.choices[0].message.content

# 加载已有的 qa_seed.json（每条 entry 已含 commentary + descriptions）
with open("qa_seed_val_2s.json", encoding="utf-8") as f:
    seeds = json.load(f)

def extract_formalized_comments(response_text: str) -> list[str]:
    lines = response_text.strip().split('\n')
    comments = []
    for line in lines:
        if line:
            comments.append(line)
    return comments

# 输出结构
training_samples = []
log_path = Path(f"log_formal/val_{args.entry}.txt")

# 主构造循环
for video in tqdm(seeds):
    vid = video["video_id"]
    task_name = video["task_name"]
    scenario_name = video["scenario_name"]
    video_entry = {
        "video_id": vid,
        "annotations": []
    }
    for entry in video["annotations"]:
        ts = entry["timestamp"]
        task_type = entry["type"]
        comments = entry["commentary"]
        descriptions = entry["descriptions"]
        goal_template = TEMPLATE_QUESTIONTYPE[task_type]

        #if task_type not in TEMPLATE_QS:
        #    print("skip")
        #    continue

        #if len(comments) < 2:
        #    print("skip")
        #    continue

        # 拼接上下文
        comment_text = " ".join(comments)
        desc_text = "\n".join(f"[{d['timestamp']:.1f}s] {d['text']}" for d in descriptions)

        # 生成专家回答 Â
        prompt_A = f"""You are an expert assistant for editing instructional commentary in skill demonstration videos. I will provide you with segments of transcribed expert commentary from videos. These comments may include informal spoken expressions (e.g., "you know...", "I guess...", "kinda", etc.) and might consist of multiple sentences that are either semantically redundant or cover different aspects of the demonstration.

#Input to be processed
{comment_text}

#Input Description:
    - Each group of comments is associated with a timestamp.
    - The comments are labeled as {task_type}.

#Output Requirements:
1. Formalization:
    - Eliminate all informal spoken language and filler expressions to produce clear, professional, and written-style language.

2. Semantic Consolidation:
    - If multiple comments convey similar or overlapping content, merge them into a single, concise paragraph.
    - If they refer to distinct points, split them into separate paragraphs, each focused on one specific point.

3. Categorical Clarity:
    - Each paragraph should address only one execution quality or improvement suggestion.
    - AVOID merging multiple suggestions into one paragraph by using transitions like "Additionally", "Also", "and", or "then".

4. Ensure Depth of Comments:
    - if the original comment are too simple or obvious, or only describe a simple action without explaining or reasoning on it (e.g., "the execution is good/effective", "she raise her left hand"), discard them.
    
5. Information Grounding:
    - DO NOT give information not mentioned in the commentary segement.
    - DO NOT add information not mentioned in the oriiginal commentary segment.

6. Ignore Meta-comments:
    - If any comment appears to reflect on the annotation process itself (e.g., apologizing for an error, noting that the expert didn’t fully review the video, expressing confusion or excitement about the video, stating unvisibility of the video), do not include it in the output, even if it is phrased in an instructional tone. Only keep comments that clearly relate to the physical performance or improvement of the task shown in the video.

7. Labeling:
    - Each output segment should labeled by [Good Execution] or [Tip for Improvement] according to the original label.

#Output Format:
[<Label>] comment1
[<Label>] comment2
..."""
        A_hat = llm_call(prompt_A)
        A_group = extract_formalized_comments(A_hat)

        for i, A in enumerate(A_group):
            prompt_Qe = f"""You are an AI assistant tasked with generating insightful questions. You will be observing a task video, presented as a "Narration" (seen as the surface-level actions) and the corresponding "Expert Commentary" (seen as deeper insights, reasons, or techniques). 

#Narration of the video (seen as the surface-level actions)
{desc_text}

#Expert commentary of the video (seen as deeper insights, reasons, or techniques)
{A}

#Goal
Your goal is to formulate questions specific on the observed actions after seeing the "Narration" to understand the deeper insights revealed in the "Expert Commentary", keeping in mind the specific context provided above. The questions should ultimately guide the learner towards an understanding similar to what an expert might articulate. The Expert Commentary is labeled as {task_type}.

#Guidelines

##Should
    1. Ask for more than what is obvious from the "Narration" alone. It should probe into the *reasons*, *intentions*, *subtle techniques*, or *critical judgments* highlighted or implied by the "Expert Commentary".
    2. The question should conceptually related to the "Expert Commentary", but not presuppose the "Expert Commentary".
    3. Based on the observed actions.
    4. {goal_template}
    
##Should NOT
    1. DO NOT ask for general evaluations, summaries, opinions or suggestions, such as:
        - "What is the overall quality of the performance?" (or other semantically similar phrases)
        - "What are the main points of the expert commentary?" (or other semantically similar phrases)
    2. AVOID presuppose a *positive or negative* relationships between two observed actions.
    3. AVOID presuppose a *positive or negative* consequences of the actions.
    4. SHOULD NOT be TOO detailed by using technical terms overlap with the "Expert Commentary"
    5. AVOID questions that are too general, contain words like:
        - "specific", "aspect", "positioning", "movement", "contribute", "help", "improve", "stability", "effectiveness", "adjust"...

##Word Restrictions
    1. AVOID vague phrases like "the video", "this movement", or "the performance".
    2. DO NOT use words like "as noted/mentioned in the expert commentary" or other similar phrases that refer to the Expert Commentary.
    3. DO NOT mention the timestamp of the video in the question.
    4. DO NOT use multiple question words in one question (such as "What ..., and how...?")

##Information Grounding
    1. DO NOT give information or ask about information not mentioned in the provided "Narration" and "Expert Commentary".

##Output format
[question] ..."""

            Qe = llm_call(prompt_Qe)
        #Q_group = extract_formalized_comments(Qe)
            Qe = re.search(r"\[question\]\s*(.+)", Qe).group(1).strip()

            # cross check phase
            #answer_withcomment = llm_call(f"""Given the video narration and the expert commentary, answer the question in brief.\nVideo narration: {desc_text}\nExpert commentary: {A}\nThe answer should not contain words "the expert", "the expert commentary", "the commentary".""")
            #answer_wocomment = llm_call(f"""Given the video narration, answer the question in brief.\nVideo narration: {desc_text}""")

            #score_w = checker.predict([(answer_withcomment, A)])
            #score_wo = checker.predict([(answer_wocomment, A)])
            #if score_w[0] <= score_wo[0]:
            #    judgement_gain = "NO"
            #else:
            #    judgement_gain = "YES"
            #score_qa = checker.predict([(A, Qe)])
            #if score_qa[0] < 3.0:
            #    judgement_qa = "NO"
            #else:
            #    judgement_qa = "YES"
            check_prompt = f"""Verify whether the generated question. The question is generated after observing a task video, presented as a "Narration" (seen as the surface-level actions) and the corresponding "Expert Commentary" (seen as deeper insights, reasons, or techniques). The question is supposed to be formulated after seeing the video to understand gain the information in the "Expert Commentary".
            
If the question follows the rules, output "OK". If not, give reasons. When generating the reason, do not mention the index of rule that is violated.

#Question
{Qe}

#Expert Commentary
{A}

#Narration
{desc_text}

#Rules
Check the rules one by one.
    1. based on the observable actions
    2. not asking for overall evaluations, summaries, opinions or suggestions
    3. not using technical terms that overlap with the "Expert Commentary"
    4. not using multiple question words in one question

#Output format
- If the question follows the rules, ouput:
OK

- If not, output the reasons:
Reason: ____"""

            check_result = llm_call(check_prompt)
            if check_result.strip() != "OK":
                reason = check_result
                #prompt_Qcheck = f"""The previous generated question "{Qe}" fail to follow the instructions due to the reason: {reason}. Please regenerate a question avoiding the failure reason above, and following the instructions below:
#{prompt_Qe}
#"""
                prompt_Qcheck = f""""You are an AI assistant tasked with generating insightful questions. You will be observing a task video, presented as a "Narration" (seen as the surface-level actions) and the corresponding "Expert Commentary" (seen as deeper insights, reasons, or techniques). 

#Narration of the video (seen as the surface-level actions)
{desc_text}

#Expert commentary of the video (seen as deeper insights, reasons, or techniques)
{A}

#Goal
Your goal is to formulate questions specific on the observed actions after seeing the "Narration" to understand the deeper insights revealed in the "Expert Commentary", keeping in mind the specific context provided above. The questions should ultimately guide the learner towards an understanding similar to what an expert might articulate. The Expert Commentary is labeled as {task_type}.

#Guidelines

##Should
    1. Ask for more than what is obvious from the "Narration" alone. It should probe into the *reasons*, *intentions*, *subtle techniques*, or *critical judgments* highlighted or implied by the "Expert Commentary".
    2. The question should conceptually related to the "Expert Commentary", but not presuppose the "Expert Commentary".
    3. Based on the observed actions.
    4. {goal_template}
    
##Should NOT
    1. DO NOT ask for general evaluations, summaries, opinions or suggestions, such as:
        - "What is the overall quality of the performance?" (or other semantically similar phrases)
        - "What are the main points of the expert commentary?" (or other semantically similar phrases)
    2. AVOID presuppose a *positive or negative* relationships between two observed actions.
    3. AVOID presuppose a *positive or negative* consequences of the actions.
    4. SHOULD NOT be TOO detailed by using technical terms overlap with the "Expert Commentary"
    5. AVOID questions that are too general, contain words like:
        - "specific", "aspect", "positioning", "movement", "contribute", "help", "improve", "stability", "effectiveness", "adjust"...

##Word Restrictions
    1. AVOID vague phrases like "the video", "this movement", or "the performance".
    2. DO NOT use words like "as noted/mentioned in the expert commentary" or other similar phrases that refer to the Expert Commentary.
    3. DO NOT mention the timestamp of the video in the question.
    4. DO NOT use multiple question words in one question (such as "What ..., and how...?")

##Information Grounding
    1. DO NOT give information or ask about information not mentioned in the provided "Narration" and "Expert Commentary".

##Bad Example and why it is bad (DO NOT generate question like this by avoiding the reason below)
[question] {Qe}
[reason for why it is bad question] {reason}

##Output format
[question] ..."""
                Qe_new = llm_call(prompt_Qcheck)
                Qe_new = re.search(r"\[question\]\s*(.+)", Qe_new).group(1).strip()
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"Timestamp: {ts}\n")
                    log_file.write(f"Formalized A:\n{A}\n")
                    log_file.write(f"Bad Qe:\n{Qe}\n")
                    #log_file.write(f"Score:\n{score_qa}\n")
                    log_file.write(f"Reason:\n{reason}\n")
                    log_file.write(f"Rewritten Qe:\n{Qe_new}\n\n")
                Qe = Qe_new
            else:
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"Timestamp: {ts}\n")
                    log_file.write(f"Formalized A:\n{A}\n")
                    log_file.write(f"Generated Qe: \n{Qe}\n\n")
                    #log_file.write(f"Score:\n{score_qa}\n")
            video_entry["annotations"].append({
                "timestamp": ts,
                "type": task_type,
                "commentary": comments,
                "A_hat": A,
                "Qe": Qe,
            })
    if video_entry["annotations"]:
        training_samples.append(video_entry)

out_path = Path(f"log_formal/qa_val_samples_{args.entry}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(training_samples, f, indent=2, ensure_ascii=False)