import os, json, re, asyncio, aiohttp, openai, tiktoken
from pathlib import Path
from collections import defaultdict
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# ---------- FILE NAME (modify the FILE_IN and FILE_OUT path) ----------
FILE_IN  = Path("log_formal/qa_train_samples.json")
FILE_OUT = Path("log_formal/qa_train_samples_cleaned.json")
MODEL    = "gpt-4o-mini"
MAX_CONCURRENCY = 20
client   = AsyncOpenAI()               # in the environment, set OPENAI_API_KEY

# --------------------
EVAL_WORDS  = re.compile(
    r"\b(effective|efficient|good|great|well|poor|bad|correct|incorrect|"
    r"sufficient|insufficient|adequate|inadequate|appropriate|inappropriate|"
    r"useful|useless|successful|unsuccessful|clear|unclear|done|complete|incomplete)\b", re.I)
CAUSE_WORDS = re.compile(
    r"\b(because|since|as|by|through|which|that|so|helps?|allow|enable|ensure|"
    r"lead[sd]?|result[sd]?|cause[sd]?|improv[esd]?|thus)\b", re.I)

# ---------- Initially remove the comments that are too short or shallow ----------
def heuristic_is_shallow(txt: str, max_len: int = 22) -> bool:
    words = txt.split()
    return (len(words) <= max_len
            and EVAL_WORDS.search(txt)
            and not CAUSE_WORDS.search(txt))

SYSTEM_PROMPT = """
You are a meticulous annotator.

TASK
----
Given a single sentence from the "A_hat" field, decide whether it
(a) merely states an evaluation (good/bad/sufficient/insufficient, etc.)
    **without** explaining *why* or *how*,              → label **REMOVE**
(b) provides an evaluation **and** offers some reason,
    mechanism, evidence, or suggestion.                → label **KEEP**

OUTPUT
------
Respond with exactly one uppercase word, either KEEP or REMOVE.
Do NOT output anything else.
""".strip()

# ---------- GPT 分类 ----------
async def classify(txt: str) -> bool:
    """
    发送到 o3，返回 True 表示应当 REMOVE。
    """
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": txt}
            ],
            max_tokens=2,
            timeout=30,
        )
        label = resp.choices[0].message.content.strip().upper()
        return label == "REMOVE"
    except Exception as e:
        print("OpenAI error:", e)
        return False

async def main():
    data = json.loads(FILE_IN.read_text(encoding="utf-8"))

    flat_records = []           # (video_idx, ann_idx, ann_dict)
    for vidx, video in enumerate(data):
        for aidx, ann in enumerate(video.get("annotations", [])):
            flat_records.append((vidx, aidx, ann))

    # initally filter out. Can be improved?
    cand = [(v, a, ann) for v, a, ann in flat_records
            if heuristic_is_shallow(ann["A_hat"])]

    # GPT
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    removals = set()            # {(video_idx, ann_idx)}

    async def run_one(rec):
        vidx, aidx, ann = rec
        async with sem:
            if await classify(ann["A_hat"]):
                removals.add((vidx, aidx))

    await tqdm.gather(*(run_one(r) for r in cand), total=len(cand))

    # write file
    for vidx, video in enumerate(data):
        video["annotations"] = [ann for aidx, ann in enumerate(video["annotations"])
                                if (vidx, aidx) not in removals]

    FILE_OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    kept = sum(len(v["annotations"]) for v in data)
    print(f"Videos: {len(data)}, total annotations: {len(flat_records)}, "
          f"removed: {len(removals)}, kept: {kept}")

if __name__ == "__main__":
    asyncio.run(main())