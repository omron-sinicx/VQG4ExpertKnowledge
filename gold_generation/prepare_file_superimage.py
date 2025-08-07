from openai import OpenAI
import base64
import os
from pathlib import Path
import json
from pathlib import Path
from tqdm import tqdm

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def call_llm(image_url, scenario_name, task_name):
    chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
                {
                    "type": "text",
                    "text": f"""
                    #Grid Image Guidance
                    The image arranges frames uniform-sampled from a video in a grid view.
                    #Reasoning Guidance
                    Your goal is to ask a question that a curious novice, after seeing the video, might ask to understand the deeper insights that an expert can provide.
                    #Video Scenario
                    This is a video of {scenario_name} scenario, and the task is {task_name}.
                    #Question Guidance
                    You should ask scene-specific question for the image.
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
                    #Format Guidance
                    Please output your question in the following format:
                    <Question>
                    """
                },
            ],
        },
    ],
    )
    return chat_response.choices[0].message.content.strip()

# Iterate through all images in the superimages folder
comm_data = json.load(open("qa_seed_with_videos.json", "r", encoding="utf-8"))
superimages_folder = "superimages"
log_path = Path("log/log_superimages_0.txt")
processed_video = []

for video in tqdm(comm_data):
    for entry in video["annotations"]:
        try:
            image_path = "superimages/"+entry["video"].split("/")[-1]+"jpg"
            scenario_name = entry["scenario_name"]
            task_name = entry["task_name"]
        except KeyError:
            print("KeyError: Missing video_path, scenario_name, or task_name in entry.")
            continue
        if image_path in processed_video:
            print(f"Skipping already processed video: {image_path}")
            continue
        else:
            processed_video.append(image_path)
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"

        # Call the LLM for each image
        response = call_llm(base64_qwen, scenario_name, task_name)
        with open(log_path, "a") as log_file:
            log_file.write(f"Image: {image_path}\n")
            log_file.write(f"Response: {response}\n\n")

#chat_response = client.chat.completions.create(
#    model="Qwen/Qwen2.5-VL-7B-Instruct",
#    messages=[
#        {"role": "system", "content": "You are a helpful assistant."},
#        {
#            "role": "user",
#            "content": [
#                {
#                    "type": "image_url",
#                    "image_url": {
#                        "url": base64_qwen1,
#                    },
#                },
#                {
#                    "type": "image_url",
#                    "image_url": {
#                        "url": base64_qwen2,
#                    },
#                },
#                {
#                    "type": "text",
#                    "text": """
#                    #Input Grid Image Guidance
#                    Each image arranges frames uniform-sampled from a video in a grid view.
#                    The two images show the scene of different clips from a same task video.
#                    #Reasoning Guidance
#                    Your goal is to ask a question that a curious novice, after seeing the video, might ask to understand the deeper insights that an expert can provide.
#                    #Question Guidance
#                    You should compare the two images.
#                    But when asking question, you should ask a scene-specific question for each image SEPERATLY.
#                    Focus on a concrete moment, gesture, or object.
#                    The answer of the question should highlight the insight that an expert can provide.
#                    #Question Restrictions
#                    DO NOT ask for general evaluations, summaries, or opinions.
#                    DO NOT directly ask what specific techniques are used.
#                    DO NOT directly ask what insights the expert can provide.
#                    Ask only ONE question.
#                    #Good Question Example
#                    - Why would a climber rub their hands together after removing them from a chalk bag?
#                    - Why does the performer adjust the heat before flipping the omelet?
#                    - Why does the performer jump off the left hand when catching the basketball with the right hand?
#                    #Format Guidance
#                    Please output your question in the following format:
#                    <Question 1>
#                    <Question 2>
#                    """
#                    #"type": "text",
#                    #"text": """
#                    #<Grid Image Guidance>
#                    #The image arranges frames uniform-sampled from a video in a grid view.
#                    #<Reasoning Guidance>
#                    #Your goal is to ask a question that a curious novice, after seeing the video, might ask to understand the deeper insights that an expert can provide.
#                    #<Question Guidance>
#                    #You should ask scene-specific question for the image.
#                    #The question should focus on a concrete action, gesture, or object.
#                    #"""
#                },
#            ],
#        },
#    ],
#)
#print("Chat response:", chat_response)