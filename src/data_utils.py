import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
DEFAULT_IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")

REQUIRED_COLUMNS = {"sample_id", "post_id", "hashtag", "label"}


def resolve_image_path(post_id, images_dir=DEFAULT_IMAGES_DIR):
    post_id = str(post_id)
    filename = post_id if post_id.lower().endswith(".png") else f"{post_id}.png"
    return os.path.join(images_dir, filename)


def load_dataframe(csv_path=DEFAULT_CSV_PATH, images_dir=DEFAULT_IMAGES_DIR, drop_missing_images=True):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"dataset.csv is missing required columns: {missing_cols}")

    label_min, label_max = df["label"].min(), df["label"].max()
    if label_min >= 1 and label_max <= 5:
        df["label"] = df["label"] - 1
    elif label_min >= 0 and label_max <= 4:
        logger.info("label already in range 0-4, skipping -1 shift.")
    else:
        raise ValueError(f"Unexpected label range: [{label_min}, {label_max}] (only 1-5 or 0-4 allowed)")

    df["image_path"] = df["post_id"].apply(lambda pid: resolve_image_path(pid, images_dir))

    exists_mask = df["image_path"].apply(os.path.exists)
    n_missing = int((~exists_mask).sum())
    if n_missing > 0:
        for _, row in df.loc[~exists_mask].iterrows():
            logger.warning(
                "Image not found, skipping: sample_id=%s post_id=%s path=%s",
                row["sample_id"], row["post_id"], row["image_path"],
            )
        logger.warning("Skipping %d row(s) due to missing images (out of %d total).", n_missing, len(df))
        if drop_missing_images:
            df = df.loc[exists_mask].reset_index(drop=True)
    else:
        logger.info("No rows with missing images (%d rows total).", len(df))

    return df
