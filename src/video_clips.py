"""
This script extracts exo and ego video clips, and the descriptions about the video clips
based on original EgoExo4D annotations and descriptions, and the EgoExoAsk QA pairs.
It use the "best exo camera" from the atomic descriptions to determine which camera to use for each clip.
This script only process the validation set.

Output file:
[
    {
        "video_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        "annotations": [
            {
                "timestamp": 1.792035,
                "A_hat": [
                    "[Good Execution] The climber demonstrates a clear understanding of technique by brushing the holds before grasping them. This action indicates an awareness of improving grip quality and preparing for secure contact with the handholds.",
                    "[Good Execution] In addition to brushing, the climber also carefully inspects the hold, showing attention to detail and preparation prior to engaging with the climbing surface."
                ],
                "Qe": [
                    "Why might brushing the climbing holds before grasping them be an intentional choice during a climb?",
                    "Why might the climber be inspecting the hold while removing chalk?"
                ],
                "descriptions": [
                    {
                        "timestamp": 0.0,
                        "text": "xxxxxxxxxxx",
                        "best_exo": "exo3"
                    },
                    {
                        "timestamp": 0.0,
                        "text": "xxxxxxxxxxx",
                        "best_exo": "exo2"
                    }
                ],
                "best_exo": "exo3",
                "video": "clips/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/00.mp4"
            },
            ....
        ]
    },
    {
        "video_id":
"""

import json
from pathlib import Path
from collections import defaultdict
from ast import literal_eval
import subprocess
import re
exo_cam = {"-1": None, "1":"exo1", "2":"exo2", "3":"exo3", "4":"exo4", "5":"exo5", "6":"exo6"}

def cal_best_exo(desc):
# desc: [{"timestamp":,... "text":..., "best_exo":["exo1", "exo2", ...]}, ...]
    best_exo_count = defaultdict(int)
    for d in desc:
        if d["best_exo"]:
            best_exo_count[d["best_exo"]] += 1
    if best_exo_count:
        return max(best_exo_count, key=best_exo_count.get)
    else:
        return "exo1"

# extract best_exo camera
desc_data = json.load(open("/workspace/data/data_egoexo/annotations/atomic_descriptions_val.json"))["annotations"] # Modify the EgoExo4D annotation path

desc_dict = defaultdict(list)

vid_root = "/workspace/data/data_egoexo" # Modify the EgoExo4D video path

comm = json.load(open("/workspace/data/data_egoexo/annotations/proficiency_demonstration_val.json"))["annotations"] # Modify the EgoExo4D annotation path
all_vid_paths = {entry["take_uid"]: entry["video_paths"]["exo1"].split("/cam")[0] for entry in comm}
all_ego_paths = {entry["take_uid"]: entry["video_paths"]["ego"] for entry in comm}
indexed_comm = { item['take_uid']: item for item in comm}

# === Build description index {vid: List[{"timestamp", "text"}]} ===
desc_dict = defaultdict(list)
for vid, annots in desc_data.items():
    for annot in annots:
        for d in annot.get("descriptions", []):
            desc_dict[vid].append({"timestamp": d["timestamp"], "text": d["text"], "best_exo": exo_cam[d["best_exo"]["raw_cam_id"]] if d.get("best_exo") else None})

qa = json.load(open("annotations/EgoExoAsk_val.json"))
merged_qa = []

window_size = 7.0  # seconds

for entry in qa:
    grouped = defaultdict(list)
    # {timestamp: [(A_hat1, Qe1), (A_hat2, Qe2), ...]}
    for ann in entry["annotations"]:
        grouped[ann["timestamp"]].append((ann.get("A_hat"), ann.get("Qe")))
        # {timestamp: [(A_hat1, Qe1), (A_hat2, Qe2), ...]}

    new_annotations = []
    for ts, qas in grouped.items():
        hats = [hat for hat, _ in qas]
        qes = [qe for _, qe in qas]
        nearby_desc = sorted(
            [d for d in desc_dict[entry["video_id"]] if 0 <= ts - d["timestamp"] <= window_size],
            key=lambda x: x["timestamp"]
        )

        best_seg_exo = cal_best_exo(nearby_desc)
        start_time = max(0.0, ts - window_size)
        for d in nearby_desc:
            d["timestamp"] = round(d["timestamp"] - start_time, 6)
        new_annotations.append({
            "timestamp": ts,
            "A_hat": hats,
            "Qe": qes,
            "descriptions": nearby_desc,
            "best_exo": best_seg_exo
        })
    merged_qa.append({
        "video_id": entry["video_id"],
        "annotations": new_annotations
    })

clips_dir = Path("./clips")
clips_dir.mkdir(parents=True, exist_ok=True)

for entry in merged_qa:
    vid = entry["video_id"]
    entry["scenario_name"] = indexed_comm[vid]["scenario_name"]
    entry["task_name"] = indexed_comm[vid]["task_name"]
    ego_path = str(Path(vid_root) / all_ego_paths[vid]) # like this: str(takes/unc_basketball_03-30-23_02_5/frame_aligned_videos/aria02_214-1.mp4)
    if not Path(ego_path).exists():
        print(f"Warning: Video path {ego_path} does not exist for video {vid}. Skipping.")
        continue

    for i, qa_item in enumerate(entry["annotations"]):
        try:
            exo_path = str(Path(vid_root) / indexed_comm[vid]["video_paths"][qa_item["best_exo"]])
        except KeyError:
            print(f"Warning: Best exo camera {qa_item['best_exo']} not found for video {vid}. Using exo1.")
            exo_path = str(Path(vid_root) / indexed_comm[vid]["video_paths"]["exo1"])
        end_time = qa_item["timestamp"]
        start_time = max(0, end_time - window_size)
        duration = end_time - start_time
        clip_name = f"{i:02d}.mp4"
        clip_name_ego = f"{i:02d}_ego.mp4"
        clip_path = clips_dir / vid / clip_name
        clip_path_ego = clips_dir / vid / clip_name_ego
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path_ego.parent.mkdir(parents=True, exist_ok=True)

        if clip_path.exists():
            print(f"Clip {clip_path} already exists, skipping extraction.")
            qa_item["video"] = str(clip_path)
            continue
        else:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", exo_path,
                "-t", str(duration),
                "-c", "copy",
                str(clip_path)
            ], stdout=subprocess.DEVNULL,
               stderr=subprocess.STDOUT,
               check=True)
            qa_item["video"] = str(clip_path)
            
        if clip_path_ego.exists():
            print(f"Clip {clip_path_ego} already exists, skipping extraction.")
            qa_item["video_ego"] = str(clip_path_ego)
            continue
        else:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", ego_path,
                "-t", str(duration),
                "-c", "copy",
                str(clip_path_ego)
            ], stdout=subprocess.DEVNULL,
               stderr=subprocess.STDOUT,
               check=True)
            qa_item["video_ego"] = str(clip_path_ego)

# Save
with open("annotations/qa_val_samples_video_w_desc_test.json", "w") as f:
    json.dump(merged_qa, f, indent=4)