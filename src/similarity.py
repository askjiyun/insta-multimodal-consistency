import logging
import os
import sys
from itertools import combinations

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import resolve_image_path  # noqa: E402
from models import build_model  # noqa: E402
from train import get_device  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_N_COMBINATIONS = 5000


@torch.no_grad()
def compute_embedding_similarities(df, backbone, processor, device, images_dir=None, log_every=200):
    backbone.eval()
    similarities = []
    kept_idx = []

    for i, (idx, row) in enumerate(df.iterrows()):
        image_path = row["image_path"] if "image_path" in row.index else resolve_image_path(row["post_id"], images_dir)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning("Image load failed, skipping: %s (%s)", image_path, e)
            continue

        text = str(row["hashtag"])
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = backbone(**inputs)

        if hasattr(outputs, "image_embeds") and hasattr(outputs, "text_embeds"):
            image_embed = outputs.image_embeds[0].cpu().numpy()
            text_embed = outputs.text_embeds[0].cpu().numpy()
        else:
            image_embed = outputs.last_hidden_state[:, 0, :][0].cpu().numpy()
            text_embed = outputs.last_hidden_state[:, 0, :][0].cpu().numpy()

        sim = cosine_similarity([image_embed], [text_embed])[0][0]
        similarities.append(sim)
        kept_idx.append(idx)

        if (i + 1) % log_every == 0:
            logger.info("Similarity computation progress: %d/%d", i + 1, len(df))

    kept_df = df.loc[kept_idx].reset_index(drop=True)
    return np.array(similarities), kept_df


def similarity_to_score_with_threshold(similarity, thresholds):
    for level, t in enumerate(thresholds):
        if similarity < t:
            return level
    return len(thresholds)


def find_optimal_thresholds(similarities, labels, n_combinations=DEFAULT_N_COMBINATIONS, seed=42):
    rng = np.random.RandomState(seed)
    candidate_thresholds = sorted(rng.uniform(0.05, 0.95, 100))
    candidate_combinations = list(combinations(candidate_thresholds, 4))

    best_f1 = -1.0
    best_thresholds = None
    for t in candidate_combinations[:n_combinations]:
        preds = [similarity_to_score_with_threshold(s, t) for s in similarities]
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = t

    logger.info("Optimal threshold(outer train)=%s, train macro-F1=%.4f", best_thresholds, best_f1)
    return best_thresholds, best_f1


def run_similarity_fold(model_key, fold, dataset_df, folds_df, images_dir=None, n_combinations=DEFAULT_N_COMBINATIONS, seed=42):
    device = get_device()
    backbone, processor = build_model(model_key)
    backbone.to(device)

    merged = dataset_df.merge(folds_df[["sample_id", "fold"]], on="sample_id", how="inner")
    outer_train_df = merged[merged["fold"] != fold].reset_index(drop=True)
    outer_test_df = merged[merged["fold"] == fold].reset_index(drop=True)

    train_sims, train_kept_df = compute_embedding_similarities(outer_train_df, backbone, processor, device, images_dir)
    thresholds, train_f1 = find_optimal_thresholds(
        train_sims, train_kept_df["label"].values, n_combinations=n_combinations, seed=seed
    )

    test_sims, test_kept_df = compute_embedding_similarities(outer_test_df, backbone, processor, device, images_dir)
    test_pred = [similarity_to_score_with_threshold(s, thresholds) for s in test_sims]

    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model_key": model_key,
        "fold": fold,
        "y_true": test_kept_df["label"].tolist(),
        "y_pred": test_pred,
        "sample_ids": test_kept_df["sample_id"].tolist(),
        "post_ids": test_kept_df["post_id"].tolist(),
        "thresholds": list(thresholds) if thresholds is not None else None,
        "train_f1": train_f1,
    }
