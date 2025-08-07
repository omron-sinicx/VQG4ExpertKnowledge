import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationMixin
import faiss
from sentence_transformers import SentenceTransformer
import json

#from transformers.configuration_utils import PretrainedConfig
#PretrainedConfig.has_no_defaults_at_init = True
# model setting
IDX     = faiss.read_index("log_formal/comment_db_zsst.index")
METAS   = json.load(open("log_formal/comment_texts_zsst.json"))
#ST_EMB  = SentenceTransformer("final/retriever_st_2/final")
ST_EMB  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_ctx(query_text, k=5):
    qv = ST_EMB.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(qv)
    _, I = IDX.search(qv, k)
    return [METAS[i]["text"] for i in I[0]]

model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().to(torch.bfloat16)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def get_num_frames_by_duration(duration):
        local_num_frames = 4        
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments
        
        num_frames = min(512, num_frames)
        num_frames = max(128, num_frames)

        return num_frames

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration = False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

from torchvision.io import read_video
from PIL import Image
import torch

def load_video_torchvision(
    video_path,
    bound=None,
    input_size=448,
    max_num=1,
    num_segments=32,
    get_frame_by_duration=False,
):
    """
    用 torchvision.io.read_video 代替 Decord。
    返回：
      • pixel_values: Tensor, shape (总patch数, C, input_size, input_size)
      • num_patches_list: List[int], 每个原始帧对应的 patch 数
    """
    # 1. 读全量视频帧
    # video: Tensor[K, H, W, C], dtype=uint8； _ : audio（忽略）； info 包含 video_fps
    video, _, info = read_video(video_path, pts_unit="sec")
    
    total_frames = video.shape[0]
    max_frame = total_frames - 1
    fps = float(info.get("video_fps", 30.0))  # 默认 30 fps 兜底
    
    # 2. 计算要抽的帧索引
    if get_frame_by_duration:
        duration = total_frames / fps
        num_segments = get_num_frames_by_duration(duration)
    frame_indices = get_index(bound, fps, max_frame,
                              first_idx=0, num_segments=num_segments)
    
    # 3. 对每帧做预处理
    pixel_values_list = []
    num_patches_list = []
    transform = build_transform(input_size=input_size)
    
    for fi in frame_indices:
        # video[fi]: Tensor[H, W, C]，先转成 numpy，再转 PIL
        frame = video[fi].numpy()
        img = Image.fromarray(frame).convert("RGB")
        
        # dynamic_preprocess 返回 List[PIL.Image]（tiles）
        tiles = dynamic_preprocess(
            img, image_size=input_size,
            use_thumbnail=True, max_num=max_num
        )
        # 每个 tile 过 transform、堆成 Tensor
        patch_ts = torch.stack([transform(t) for t in tiles])
        
        num_patches_list.append(patch_ts.shape[0])
        pixel_values_list.append(patch_ts)
    
    # 把所有 patch 在第 0 维拼接
    pixel_values = torch.cat(pixel_values_list, dim=0)
    return pixel_values, num_patches_list

# evaluation setting
max_num_frames = 512
generation_config = dict(
    do_sample=True,
    temperature=0.6,
    max_new_tokens=50,
    top_p=0.95,
)


from pathlib import Path
from tqdm import tqdm
comm_data = json.load(open("log_formal/qa_val_samples_video_w_desc_eval.json", "r", encoding="utf-8"))
log_path = Path(f"log_formal/qa_log_internvl_exo_rag.txt")
training_samples = []

for video in tqdm(comm_data):
    video_entry = {
        "video_id": video["video_id"],
        "annotations": [],
    }
    scenario_name = video["scenario_name"]
    task_name = video["task_name"]
    for entry in video["annotations"]:
        try:
            video_path = entry["video"]
        except KeyError:
            print(f"Skipping due to missing video path.")
            continue

        descriptions = entry.get("description", "")
        desc_text = "\n".join(f"[{d['timestamp']:.1f}s] {d['text']}" for d in descriptions)

        ctx_list = retrieve_ctx(desc_text, k=5)
        ctx_block  = "\n".join(f"- {c}" for c in ctx_list)

        num_segments=128

        with torch.no_grad():
    
            pixel_values, num_patches_list = load_video_torchvision(video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=False)
            pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
            question1 = f"""You are an AI assistant tasked with generating insightful questions. You will be observing a task video.

#Goal
Your goal is to formulate ONLY ONE question specific on the video after seeing the video to understand the deeper insights that an  expert of this task will provide about the video.

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
    5. DO NOT mention the existance of other clips.
    
##Context from other clips
Following are the experts' comments from other clips (not for the current clip, but similar to the current clip) which may help you to generate a more insightful question.
{ctx_block}

##Output format
ONLY output one question.
        """
            question = video_prefix + question1
            output1, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
            output1 = output1.strip()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Video ID: {video['video_id']}\n")
                f.write(f"Question: {output1}\n")
                f.write("\n")
            video_entry["annotations"].append({
                "question": output1,
                "video": video_path,
            })
    if video_entry["annotations"]:
        training_samples.append(video_entry)

out_path = Path("log_formal/qa_internvl_exo_rag.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(training_samples, f, indent=2, ensure_ascii=False)


            


        

