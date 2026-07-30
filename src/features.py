import logging
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import resolve_image_path  # noqa: E402

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_features(df, backbone, processor, device, images_dir=None, log_every=500):
    backbone.eval()
    feats = []
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

        combined = torch.cat((outputs.image_embeds[0], outputs.text_embeds[0]), dim=0)
        feats.append(combined.cpu().numpy())
        kept_idx.append(idx)

        if (i + 1) % log_every == 0:
            logger.info("Feature extraction progress: %d/%d", i + 1, len(df))

    X = np.stack(feats, axis=0)
    kept_df = df.loc[kept_idx].reset_index(drop=True)
    return X, kept_df
