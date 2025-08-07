import av
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = LlavaNextVideoProcessor.from_pretrained(model_id)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
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


# define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image", "video") 

prompt = f"""You are an AI assistant tasked with generating insightful questions. You will be observing a task video.

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
    """
conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video"},
            ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

video_path = "clips/0e0dfbca-9593-427e-9d7a-beee8aea3aa3/04.mp4"
container = av.open(video_path)

# sample uniformly 8 frames from the video, can sample more for longer videos
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 16).astype(int)
clip = read_video_pyav(container, indices)
inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
a = processor.decode(output[0][2:], skip_special_tokens=True)
a = a.split("ASSISTANT:")[-1].strip()