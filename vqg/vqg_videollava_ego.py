import av
import numpy as np
import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf").to("cuda")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")


import json
from pathlib import Path
from tqdm import tqdm
comm_data = json.load(open("log_formal/qa_val_samples_5_video_w_desc_eval.json", "r", encoding="utf-8"))
log_path = Path(f"log_formal/log_llavavideo_ego.txt")
training_samples = []

for video in tqdm(comm_data):
    video_entry = {
        "video_id": video["video_id"],
        "annotations": [],
    }
    for entry in video["annotations"]:
        try:
            video_path = entry["video_ego"]
        except KeyError:
            continue
        with torch.no_grad():
            num_segments = 8
            container = av.open(video_path)
        
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / num_segments).astype(int)
            clip = read_video_pyav(container, indices)

            prompt = f"""USER: <video>You are an AI assistant tasked with generating insightful questions. You will be observing a task video.

#Goal
Your goal is to formulate ONLY ONE question specific on the video after seeing the video to understand the deeper insights that an expert of this task will provide about the video.

##Should
    1. Ask for more than what is obvious from the video alone. It should probe into the *reasons*, *intentions*, *subtle techniques*, or *critical judgments* that an expert may provide.
    2. Based on the observed actions.
    3. Focus on subtle actions or movements and consider whether the actions are critical in the task.

##Should not
    1. DO NOT ask for general evaluations, summaries, opinions or suggestions, such as:
        - "What is the overall quality of the performance?" (or other semantically similar phrases)
        - "What are the main points of the expert commentary?" (or other semantically similar phrases)
    2. DO NOT use multiple question words in one question (such as "Why is  ..., and how does...?")
    3. DO NOT use words like "according to the expert" or other similar phrases.
    4. DO NOT use words like "in the video" or other similar phrases.

##Output format
ONLY output one question.
ASSISTANT:"""

            inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)

        # Generate
            generate_ids = model.generate(**inputs, max_length=10000)
            a = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            a = a.split("ASSISTANT:")[-1].strip()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Video ID: {video_path}\n")
            f.write(f"Question: {a}\n")
            f.write("\n")
        video_entry["annotations"].append({
            "question": a,
            "video": video_path,
        })
    training_samples.append(video_entry)

out_path = Path("log_formal/qa_videollava_ego.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(training_samples, f, indent=2, ensure_ascii=False)
