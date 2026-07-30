"""GPT-4o-mini Full Labeling

Fixed settings
- Model: gpt-4o-mini
- temperature=0.0 for reproducibility
- ANNOTATION_GUIDELINE (5-step decision procedure + boundary rules)
- Call unit: post (1 image + N tags) -> results saved as one row per pair

Safeguards
- Immediate per-item save (flush) -- no data loss on interruption
- Resumable -- re-running the script skips already-labeled posts
- Retry with exponential backoff (handles 429)
- Failed items logged to a separate CSV
"""

import os
import ast
import csv
import time
import random
import base64
from datetime import datetime
from typing import List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


# ── 1. Imports & Configuration ──────────────────────────────────────────

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Could not read OPENAI_API_KEY from .env")
client = OpenAI(api_key=api_key.strip())

CONFIG = {
    "model":          "gpt-4o-mini",
    "temperature":    0.0,
    "prompt_version": "v1_prompt",
    "seed":           42,
}

CSV      = "insta_data.csv"                 # raw data, per post
IMG_DIR  = "img_data"
OUT_CSV  = "labels_data.csv"                    # full labeling results
FAIL_CSV = "labels_failed.csv"

SLEEP    = 0.4        # delay between calls (bump to 1.0 if 429s are frequent)

print("Model:", CONFIG["model"], "| Prompt:", CONFIG["prompt_version"])


# ── 2. Load Full Target Set ─────────────────────────────────────────────

def parse_tags(v):
    s = str(v).strip()
    if s in ("", "[]", "nan"):
        return []
    try:
        return [t for t in ast.literal_eval(s) if str(t).strip()]
    except Exception:
        return []


df = pd.read_csv(CSV)
df["taglist"] = df["tags"].apply(parse_tags)
targets = df[df["taglist"].map(len) > 0].copy().reset_index(drop=True)

print(f"Posts to label      : {len(targets)}")
print(f"Expected pair count : {targets['taglist'].map(len).sum()}")
print(f"Avg tags per post    : {targets['taglist'].map(len).mean():.1f}")
print(f"Files in image folder: {len(os.listdir(IMG_DIR))}")
print("\nPost count by keyword:")
print(targets.groupby("query_hashtag").size().to_string())


# ── 3. ANNOTATION_GUIDELINE ──────────────────────────────────────────────

ANNOTATION_GUIDELINE = """
You are an expert annotator evaluating the semantic relevance between Instagram images and hashtags.

Your task:
Given an Instagram image, its caption/content, and a list of hashtags, evaluate how relevant each hashtag is to the image.

Important rules:
1. Evaluate only the hashtags provided by the user.
2. Prioritize visual evidence from the image.
3. Use the caption only as secondary context.
4. Do not overestimate relevance based only on general popularity or vague association.
5. If the visual evidence is unclear, assign a conservative score.
6. Provide all explanations in Korean.
7. The `score` field MUST be an integer between 1 and 5 (1, 2, 3, 4, or 5). Never output text, words, decimals, or ranges in the `score` field.
8. Korean slang and affective expressions (e.g., #존맛, #JMT, #소확행) must be interpreted by their idiomatic meaning, not literally.

Scoring guideline:
Score 1: 거의 관련 없음.
- The hashtag is not visually supported by the image.
- The hashtag is unrelated or only very weakly inferable.

Score 2: 낮은 관련성.
- The hashtag has a weak or indirect relation to the image.
- It requires strong contextual inference from the caption rather than visual evidence.

Score 3: 중간 관련성.
- The hashtag is somewhat related to the image.
- It may describe a partial element, broad mood, or secondary context, but not the main visual subject.

Score 4: 높은 관련성.
- The hashtag is clearly related to a major object, place, activity, mood, or visual theme in the image.

Score 5: 매우 높은 관련성.
- The hashtag directly describes the central object, main place, main activity, or dominant visual meaning of the image.

Special cases:
- Very generic hashtags should not receive high scores unless the image strongly supports them.
- Place hashtags should receive high scores only when the image provides clear place-related evidence or the visual content strongly matches the place context.
- Emotion or mood hashtags should be judged based on facial expression, color tone, composition, activity, or atmosphere.
- Brand, campaign, or promotional hashtags should be scored conservatively unless clearly visible or strongly supported.

For each hashtag, provide:
1. score (integer 1-5 only)
2. visual_evidence (in Korean)
3. reason (in Korean)
"""


class HashtagEval(BaseModel):
    hashtag: str = Field(
        description="The hashtag being evaluated, exactly as provided by the user"
    )
    score: int = Field(
        ge=1, le=5,
        description="Relevance score as an INTEGER from 1 to 5 only. "
                    "1=거의 관련 없음, 2=낮은 관련성, 3=중간 관련성, "
                    "4=높은 관련성, 5=매우 높은 관련성. "
                    "Never output text, decimals, or ranges."
    )
    visual_evidence: str = Field(
        description="Specific visual elements actually observable in the image that support "
                    "this judgment (objects, place cues, colors, expressions, activity). "
                    "Write in Korean. If no visual support exists, state that explicitly."
    )
    reason: str = Field(
        description="Concise justification for the assigned score, in Korean. "
                    "Base it on visual evidence, not on caption text alone."
    )


class AnalysisResult(BaseModel):
    image_description: str = Field(
        description="Objective description of what is visible in the image, in Korean"
    )
    hashtag_eval: List[HashtagEval] = Field(
        description="Evaluation for EVERY hashtag provided by the user, and only those. "
                    "Do not add, omit, or modify hashtags."
    )


print("Guideline length:", len(ANNOTATION_GUIDELINE), "chars")


# ── 4. Labeling Function (Retry & 429 Handling) ──────────────────────────

failed_rows = []


def encode_image(path):
    with open(path, "rb") as fp:
        return base64.b64encode(fp.read()).decode("utf-8")


def label_post(content, taglist, image_filename, max_retry=5):
    path = os.path.join(IMG_DIR, image_filename)
    if not os.path.isfile(path):
        failed_rows.append({"image_filename": image_filename, "error": "image not found"})
        return None

    b64 = encode_image(path)
    caption = content if pd.notna(content) and str(content).strip() else "(none)"
    user_text = (f"[Post text]\n{caption}\n\n"
                 f"[Hashtags to evaluate]\n" + "\n".join(taglist))

    for attempt in range(max_retry):
        try:
            resp = client.beta.chat.completions.parse(
                model=CONFIG["model"],
                messages=[
                    {"role": "system", "content": ANNOTATION_GUIDELINE},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]},
                ],
                temperature=CONFIG["temperature"],
                response_format=AnalysisResult,
            )
            return resp.choices[0].message.parsed
        except Exception as e:
            if attempt == max_retry - 1:
                failed_rows.append({"image_filename": image_filename, "error": str(e)[:300]})
                return None
            time.sleep(2 ** attempt + random.random())   # exponential backoff (handles 429)
    return None


print("Ready")


# ── 5. Run Full Labeling ─────────────────────────────────────────────────
# If interrupted, re-run this section -- already-labeled posts are skipped
# automatically.

RUN_ID = f"full_{datetime.now():%Y%m%d_%H%M}"

# Resume: check already-processed posts
done = set()
if os.path.exists(OUT_CSV):
    done = set(pd.read_csv(OUT_CSV)["image_filename"])
    print(f"Already labeled: {len(done)} -> processing the rest")

todo = targets[~targets["image_filename"].isin(done)]
print(f"Posts to process this run: {len(todo)}\n")

write_header = not os.path.exists(OUT_CSV)
f = open(OUT_CSV, "a", newline="", encoding="utf-8-sig")
w = csv.writer(f)
if write_header:
    w.writerow(["run_id", "prompt_version", "model", "temperature",
                "query_hashtag", "image_filename", "content", "image_description",
                "hashtag", "score", "visual_evidence", "reason"])

t0 = time.time()
ok = 0
try:
    for n, (_, row) in enumerate(todo.iterrows(), 1):
        res = label_post(row["content"], row["taglist"], row["image_filename"])
        if res is None:
            print(f"[{n}/{len(todo)}] FAILED {row['image_filename']}")
            continue
        for ev in res.hashtag_eval:
            w.writerow([RUN_ID, CONFIG["prompt_version"], CONFIG["model"], CONFIG["temperature"],
                        row["query_hashtag"], row["image_filename"], row["content"],
                        res.image_description, ev.hashtag, ev.score,
                        ev.visual_evidence, ev.reason])
        f.flush()
        ok += 1
        if n % 25 == 0 or n == len(todo):      # progress every 25 items
            el = time.time() - t0
            eta = el / n * (len(todo) - n)
            print(f"[{n}/{len(todo)}] {ok} done | elapsed {el/60:.1f}min | ETA {eta/60:.0f}min")
        time.sleep(SLEEP)
finally:
    f.close()

if failed_rows:
    pd.DataFrame(failed_rows).to_csv(FAIL_CSV, index=False, encoding="utf-8-sig")
    print(f"\nFailed: {len(failed_rows)} -> {FAIL_CSV}")

print(f"\nDone: {ok} processed | total {(time.time()-t0)/60:.1f}min -> {OUT_CSV}")


# ── 6. Score Distribution ────────────────────────────────────────────────

lab = pd.read_csv(OUT_CSV)
print(f"Total pairs : {len(lab):,}")
print(f"Posts       : {lab['image_filename'].nunique():,}")
print(f"Unique tags : {lab['hashtag'].nunique():,}")

print("\n=== Score Distribution ===")
vc = lab["score"].value_counts().sort_index()
for k, v in vc.items():
    print(f"  score {k}: {v:5,d} ({v/len(lab)*100:5.1f}%)  " + "█" * int(v/len(lab)*60))

print("\n=== Mean Score by Keyword (ascending) ===")
print(lab.groupby("query_hashtag")["score"].agg(["mean", "count"]).round(2)
         .sort_values("mean").to_string())


# ── 7. Label Quality Check (Samples) ─────────────────────────────────────

for s in [1, 3, 5]:
    print(f"\n{'='*72}\nscore {s} examples\n{'='*72}")
    for _, r in lab[lab["score"] == s].head(3).iterrows():
        print(f"  tag      : {r['hashtag']}")
        print(f"  image    : {str(r['image_description'])[:65]}")
        print(f"  evidence : {str(r['visual_evidence'])[:65]}")
        print(f"  reason   : {str(r['reason'])[:65]}\n")
