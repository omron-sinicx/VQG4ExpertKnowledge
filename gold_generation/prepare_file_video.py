import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from qwen_vl_utils import process_vision_info


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

TEMPLATE_QS = {
    "good_executions": "What exectuion is good, or what makes this a good execution?",
    "tips_for_improvement": "What can be done to improve the execution, or what makes this a suboptimal execution?",
}
TEMPLATE_ANSWERTYPE = {
    "good_executions": "The expert commentary focuses on the performer's good execution or why the execution is good.",
    "tips_for_improvement": "The expert commentary focuses on the suggestions given to improve the execution or the problems about the execution.",
}
TEMPLATE_QUESTIONTYPE = {
    "good_executions": "Your question must avoid stating that the execution was good, even if the commentary implies it. Instead, ask in a way that lets the answer reveal the positive aspect.",
    "tips_for_improvement": "Your question must avoid stating that the execution needs improvement, even if the commentary implies it. Instead, ask in a way that lets the answer reveal tips.",
}

processed_video = []

def prepare_prompt(entry, comment_text):
    try:
        video_path = entry["video"]+"mp4"
        print(video_path)
        scenario_name = entry["scenario_name"]
        task_name = entry["task_name"]
    except KeyError:
        return None
    if video_path in processed_video:
        return 1
    else:
        processed_video.append(video_path)

    task_type = entry["type"]
    a_type = TEMPLATE_ANSWERTYPE[task_type]
    q_type = TEMPLATE_QUESTIONTYPE[task_type]
    prompt_Qe = f"""
    #Reasoning Guidance
    Your goal is to ask a question that a curious novice, after seeing the video, might ask to understand the deeper insights that an expert can provide.
    #Video Scenario
    This is a video of {scenario_name} scenario, and the task is {task_name}.
    #Question Guidance
    You should ask scene-specific question for the video.
    Focus on a concrete moment, gesture, or object.
    The answer of the question should highlight the insight that an expert can provide.
    #Question Restrictions
    DO NOT ask for general evaluations, summaries, or opinions.
    DO NOT directly ask what specific techniques are used.
    DO NOT directly ask what insights the expert can provide.
    #Good Question Example
    - Why would a climber rub their hands together after removing them from a chalk bag?
    - Why does the performer adjust the heat before flipping the omelet?
    - Why does the performer jump off the left hand when catching the basketball with the right hand?
    """
    #prompt_Qe = f"""You are an AI assistant tasked with generating insightful questions. You will be observing a task video.
#
    #    Your goal is to formulate a question that a curious novice, after seeing the video, might ask to understand the deeper insights that an expert can provide.
    #    
    #    Ask a question based only on the video.
    #    
    #    Here are the guidelines for crafting the question:
#
    #    1.  **Perspective**: The question should be from the perspective of someone who does *not* have expert knowledge but is eager to learn the underlying expert reasoning.
    #    2.  **High-Quality & Depth**: A high-quality question should:
    #        * Focus on a concrete moment, gesture, or object described in the video.
    #        * The question should probe into the *reasons*, *intentions* or *subtle techniques*.
    #        * Essentially, the answer to the question should highlight the insight that an expert can provide.
    #    4.  **Clarity and Specificity**:
    #        * AVOID vague phrases like “the video,” “this movement,” or “the performance.”
    #        * DO NOT ask for general evaluations, summaries, or opinions.
    #        * DO NOT directly ask what specific techniques are used in the video.
    #        * Each question should focus on a concrete moment, gesture, or object in the scene.
    #        * DO NOT judge the action of the video,like:
    #            - Why is it important ..?
    #            - Why is it beneficial ..?
    #            - How does .. contribute to ..?
    #            - Why is .. considered a good technique?
    #            - How can ... be improved?
    #            - Why does ... not good?
    #            - How does ... affect ..
    #    5.  **Information Grounding**:
    #        * DO NOT give information or ask about information not mentioned in the provided video.
    #    """
    video_messages = [
        {"role": "system", "content": "You are an AI assistant tasked with generating insightful questions."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_Qe},
            {
                "type": "video",
                "video": video_path,
                "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 2, 
                'fps': 1.0  # The default value is 2.0, but for demonstration purposes, we set it to 3.0.
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
comm_data = json.load(open("qa_seed_with_videos.json", "r", encoding="utf-8"))
log_path = Path(f"log/log_video_7B_1.txt")
training_samples = []

for video in tqdm(comm_data):
    video_entry = {
        "video_id": video["video_id"],
        "annotations": [],
    }
    for entry in video["annotations"]:
        comments = entry["commentary"]
        #if len(comments) < 2:
        #    print("skip")
        #    continue
        comment_text = " ".join(comments)
        video_messages = prepare_prompt(entry, comment_text)
        if video_messages is None:
            print(f"Skipping video {entry['video']} due to missing video path.")
            continue
        if video_messages == 1:
            print(f"Skipping video {entry['video']} as it has been processed already.")
            continue
        video_messages, video_kwargs = prepare_message_for_vllm(video_messages)
        chat_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=video_messages,
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
            "answer": comment_text,
            "clipid": entry["video"],
        })
    if video_entry["annotations"]:
        training_samples.append(video_entry)

#out_path = Path(f"qa_training_samples_from_seed_with_videos_entry_1.json")
#with open(out_path, "w", encoding="utf-8") as f:
#    json.dump(training_samples, f, indent=2, ensure_ascii=False)