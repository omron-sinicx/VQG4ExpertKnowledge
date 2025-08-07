# preprocess the atomic descriptions and proficiency demonstration data so that they can be used to generate goad QA
# the commentary data is grouped by 2-second windows.
# The descriptions are matched to the commentary based on timestamps, determined by a n-second window
import json
from pathlib import Path
from collections import defaultdict
from ast import literal_eval

def parse_json(split="train"):
    # === Load data ===
    desc_data = json.load(open(f"/workspace/data/data_egoexo/annotations/atomic_descriptions_{split}.json"))["annotations"]
    comm_data = json.load(open(f"/workspace/data/data_egoexo/annotations/proficiency_demonstration_{split}.json"))["annotations"]

    indexed_comm = { item['take_uid']: item for item in comm_data}

    # === Build description index {vid: List[{"timestamp", "text"}]} ===
    desc_dict = defaultdict(list)
    for vid, annots in desc_data.items():
        for annot in annots:
            for d in annot.get("descriptions", []):
                desc_dict[vid].append({"timestamp": d["timestamp"], "text": d["text"]})

    # === Build {vid: {("timestamp", "type"): [comment_texts]}} grouped by 2-second window ===
    comm_grouped = defaultdict(lambda: defaultdict(list))

    for entry in comm_data:
        vid = entry["take_uid"]
        for field, tag in [
            ("good_executions", "good_executions"),
            ("tips_for_improvement", "tips_for_improvement")
        ]:
            # collect (ts, comment) pairs
            events = []
            for item in entry.get(field, []):
                ts = item["video_time"]
                try:
                    comments = literal_eval(item["list"]) if isinstance(item["list"], str) else item["list"]
                    for c in comments:
                        c_str = c.strip()
                        if c_str:
                            events.append((ts, c_str))
                except:
                    continue

            events.sort(key=lambda x: x[0])

            # group comments
            i = 0
            while i < len(events):
                start_ts = events[i][0]
                last_ts = start_ts
                cluster = [events[i][1]]
                j = i + 1
                while j < len(events) and events[j][0] - start_ts <= 2.0:
                    last_ts = events[j][0]
                    cluster.append(events[j][1])
                    j += 1

                comm_grouped[vid][(last_ts, tag)].extend(cluster)
                i = j

    # === Aggregate output format {video_id, annotations: [...]} ===
    WINDOW = 7.0
    all_outputs = []

    for vid in comm_grouped:
        video_entry = {
            "video_id": vid,
            "task_name": indexed_comm[vid]["task_name"],
            "scenario_name": indexed_comm[vid]["scenario_name"],
            "annotations": []
        }

        for (ts, tag), comments in comm_grouped[vid].items():
            nearby_desc = sorted(
                [d for d in desc_dict[vid] if 0 <= ts - d["timestamp"] <= WINDOW],
                key=lambda x: x["timestamp"]
            )
            if not nearby_desc:
                continue

            video_entry["annotations"].append({
                "timestamp": ts,
                "type": tag,
                "commentary": comments,
                "descriptions": nearby_desc
            })

        if video_entry["annotations"]:
            all_outputs.append(video_entry)

    # === Save ===
    with open(f"qa_seed_{split}.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parse_json("train")
    parse_json("val")