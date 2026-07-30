#!/usr/bin/env python
"""Entry point that runs a single model through 5-fold CV (spec §2, §11).

Example usage:
    python run_cv.py --model koclip_ft
    python run_cv.py --model clip_frozen --max-epochs 3 --subset-posts 100   # smoke test
"""

import argparse
import json
import logging
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from classical import run_classical_fold  # noqa: E402
from data_utils import DEFAULT_CSV_PATH, DEFAULT_IMAGES_DIR, load_dataframe  # noqa: E402
from evaluate import fold_checkpoint_dir, model_output_dir, save_model_results  # noqa: E402
from features import extract_features  # noqa: E402
from models import MODEL_REGISTRY, TRAIN_PROTOCOL, build_model  # noqa: E402
from similarity import DEFAULT_N_COMBINATIONS, run_similarity_fold  # noqa: E402
from train import get_device, train_fold  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEFAULT_FOLDS_PATH = os.path.join(PROJECT_ROOT, "data", "folds.csv")
N_FOLDS = 5


def make_smoke_subset(dataset_df, folds_df, n_posts, seed=42, rare_label=0, min_rare_posts=5):
    """Small subset for smoke testing. Rare-class (label 0) posts are prioritized,
    and fold assignment is inherited from the original folds.csv, so the overlap=0
    invariant remains intact."""
    rng = np.random.RandomState(seed)
    merged = dataset_df.merge(folds_df[["sample_id", "fold"]], on="sample_id", how="inner")

    all_posts = merged["post_id"].unique()
    rare_posts = merged.loc[merged["label"] == rare_label, "post_id"].unique()

    n_rare = min(len(rare_posts), max(min_rare_posts, n_posts // 10))
    rare_selected = rng.choice(rare_posts, size=n_rare, replace=False) if n_rare > 0 else np.array([])

    remaining_pool = np.setdiff1d(all_posts, rare_selected)
    n_remaining = max(0, n_posts - len(rare_selected))
    n_remaining = min(n_remaining, len(remaining_pool))
    remaining_selected = rng.choice(remaining_pool, size=n_remaining, replace=False)

    selected_posts = set(rare_selected.tolist()) | set(remaining_selected.tolist())

    subset_dataset_df = dataset_df[dataset_df["post_id"].isin(selected_posts)].reset_index(drop=True)
    subset_folds_df = folds_df[folds_df["post_id"].isin(selected_posts)].reset_index(drop=True)

    logger.info(
        "Smoke subset: posts=%d (rare-label=%d posts=%d) samples=%d, per-fold distribution=%s",
        len(selected_posts), rare_label, len(rare_selected), len(subset_dataset_df),
        subset_folds_df["fold"].value_counts().sort_index().to_dict(),
    )
    return subset_dataset_df, subset_folds_df


def save_env_info(path):
    import sklearn
    import torch
    import transformers

    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "device": str(get_device()),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info


def run_cv(
    model_key,
    dataset_df,
    folds_df,
    outputs_dir=DEFAULT_OUTPUTS_DIR,
    max_epochs=None,
    n_combinations=None,
    cleanup_checkpoints=False,
    images_dir=None,
):
    cfg = MODEL_REGISTRY[model_key]
    out_dir = model_output_dir(outputs_dir, model_key)
    os.makedirs(out_dir, exist_ok=True)
    save_env_info(os.path.join(out_dir, "env_info.json"))

    # svm/xgboost use frozen-backbone features independent of label/fold, so instead
    # of re-extracting per fold, features are extracted once for the whole dataset
    # and reused across all 5 folds.
    precomputed_features = None
    if cfg["type"] in ("svm", "xgboost"):
        precomputed_features = _extract_features_once(model_key, dataset_df, images_dir)

    fold_results = []
    for fold in range(N_FOLDS):
        t0 = time.time()
        if cfg["type"] == "similarity":
            result = run_similarity_fold(
                model_key, fold, dataset_df, folds_df,
                images_dir=images_dir,
                n_combinations=n_combinations or DEFAULT_N_COMBINATIONS,
            )
        elif cfg["type"] in ("svm", "xgboost"):
            X, feat_df = precomputed_features
            fold_dir = fold_checkpoint_dir(outputs_dir, model_key, fold)
            result = run_classical_fold(model_key, fold, X, feat_df, folds_df, output_dir=fold_dir)
        else:
            fold_dir = fold_checkpoint_dir(outputs_dir, model_key, fold)
            result = train_fold(
                model_key, fold, dataset_df, folds_df,
                output_dir=fold_dir, max_epochs=max_epochs,
                images_dir=images_dir, cleanup_checkpoint=cleanup_checkpoints,
            )
        elapsed = time.time() - t0
        logger.info("[%s] fold %d complete (%.1fs, test n=%d)", model_key, fold, elapsed, len(result["y_true"]))
        fold_results.append(result)

    metrics_df, oof_df = save_model_results(model_key, fold_results, outputs_dir)
    return metrics_df, oof_df, fold_results


def _extract_features_once(model_key, dataset_df, images_dir):
    import torch

    device = get_device()
    backbone, processor = build_model(model_key)
    backbone.to(device)

    t0 = time.time()
    X, feat_df = extract_features(dataset_df, backbone, processor, device, images_dir=images_dir)
    logger.info("[%s] feature extraction complete (%.1fs, n=%d, dim=%d)", model_key, time.time() - t0, *X.shape)

    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return X, feat_df


def main():
    parser = argparse.ArgumentParser(description="Run a single model through 5-fold grouped CV")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--folds-path", default=DEFAULT_FOLDS_PATH)
    parser.add_argument("--outputs-dir", default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--max-epochs", type=int, default=None, help="default: TRAIN_PROTOCOL[max_epochs]=20")
    parser.add_argument("--n-combinations", type=int, default=None, help="number of similarity threshold search combinations (default 5000)")
    parser.add_argument("--cleanup-checkpoints", action="store_true")
    parser.add_argument("--subset-posts", type=int, default=None, help="smoke test: subsample to N posts")
    parser.add_argument("--subset-seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    dataset_df = load_dataframe(args.csv_path, args.images_dir)
    folds_df = pd.read_csv(args.folds_path)

    if args.subset_posts:
        dataset_df, folds_df = make_smoke_subset(dataset_df, folds_df, args.subset_posts, seed=args.subset_seed)

    run_cv(
        args.model, dataset_df, folds_df,
        outputs_dir=args.outputs_dir,
        max_epochs=args.max_epochs,
        n_combinations=args.n_combinations,
        cleanup_checkpoints=args.cleanup_checkpoints,
        images_dir=args.images_dir,
    )


if __name__ == "__main__":
    main()
