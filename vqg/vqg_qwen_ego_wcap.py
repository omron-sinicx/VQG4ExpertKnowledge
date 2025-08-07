import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from qwen_vl_utils import process_vision_info
from torchvision.io import read_video


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

processed_video = []

def prepare_prompt(entry,scenario_name, task_name):
    try:
        video_path = entry["video_ego"]
    except KeyError:
        return None
    else:
        processed_video.append(video_path)

    prompt_Qe = f"""You are an AI assistant tasked with generating insightful questions. You will be observing a task video.

#Goal
Your goal is to formulate ONLY ONE question specific on the observed actions after seeing the video to understand the deeper insights that an expert of this task will provide.

##Should
    1. Ask for more than what is obvious from the video alone. It should probe into the *reasons*, *intentions*, *subtle techniques*, or *critical judgments* that an expert may provide.
    2. Based on the observed actions.
    3. Focus on subtle actions and consider whether the actions are critical to the task's success or failure.

##Should not
    1. DO NOT ask for general evaluations, summaries, opinions or suggestions, such as:
        - "What is the overall quality of the performance?" (or other semantically similar phrases)
        - "What are the main points of the expert commentary?" (or other semantically similar phrases)
    2. DO NOT use multiple question words in one question (such as "Why is  ..., and how does...?")

##Output format
ONLY output one question.
    """

    descriptions = entry["descriptions"]
    desc_text = "\n".join(f"[{d['timestamp']:.1f}s] {d['text']}" for d in descriptions)
    prompt_withNarration = f"""You are an AI assistant tasked with generating insightful questions. You will be observing a task video. You will also be provided with a narration of the video with timestamp, which may include descriptions of actions.

#Narration
{desc_text}

#Goal
Your goal is to formulate ONLY ONE question specific on the observed actions after seeing the video to understand the deeper insights that an expert of this task will provide.

##Should
    1. Ask for more than what is obvious from the video alone. It should probe into the *reasons*, *intentions*, *subtle techniques*, or *critical judgments* that an expert may provide.
    2. Based on the observed actions.

##Should not
    1. DO NOT ask for general evaluations, summaries, opinions or suggestions, such as:
        - "What is the overall quality of the performance?" (or other semantically similar phrases)
        - "What are the main points of the expert commentary?" (or other semantically similar phrases)
    2. DO NOT use multiple question words in one question (such as "Why is  ..., and how does...?")

##Output format
ONLY output one question.
    """

    video_messages = [
        {"role": "system", "content": "You are an AI assistant tasked with generating insightful questions."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_withNarration},
            {
                "type": "video",
                "video": video_path,
                "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 2, 
                'fps': 2.0
            }]
        },
    ]
    return video_messages


def prepare_message_for_vllm(content_messages):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}


import json
from pathlib import Path
from tqdm import tqdm
comm_data = json.load(open("log_formal/qa_val_samples_video_w_desc_eval.json", "r", encoding="utf-8"))
log_path = Path(f"log_formal/log_video_ego_wcap.txt")
training_samples = []

for video in tqdm(comm_data):
    video_entry = {
        "video_id": video["video_id"],
        "annotations": [],
    }
    scenario_name = video["scenario_name"]
    task_name = video["task_name"]
    for entry in video["annotations"]:
        video_messages = prepare_prompt(entry, scenario_name, task_name)
        if video_messages is None:
            print(f"Skipping due to missing video path.")
            continue
        video_messages, video_kwargs = prepare_message_for_vllm(video_messages)
        chat_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=video_messages,
            max_tokens=50,
            temperature=0.6,
            top_p=0.95,
            extra_body={
                "mm_processor_kwargs": video_kwargs
            }
        )
        response = chat_response.choices[0].message.content
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Video ID: {entry['video']}\n")
            f.write(f"Question: {response}\n")
            f.write("\n")
        video_entry["annotations"].append({
            "question": response,
            "video": entry["video_ego"],
        })
    if video_entry["annotations"]:
        training_samples.append(video_entry)

out_path = Path(f"log_formal/qa_ego_wcap.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(training_samples, f, indent=2, ensure_ascii=False)