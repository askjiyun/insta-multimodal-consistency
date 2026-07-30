import logging
import os
import sys

from PIL import Image
from torch.utils.data import Dataset
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import DEFAULT_IMAGES_DIR, resolve_image_path  # noqa: E402

logger = logging.getLogger(__name__)


class ClipHashtagDataset(Dataset):
    def __init__(self, dataframe, processor, images_dir=DEFAULT_IMAGES_DIR, max_length=77):
        self.data = dataframe.reset_index(drop=True)
        self.processor = processor
        self.images_dir = images_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image_path"] if "image_path" in row else resolve_image_path(row["post_id"], self.images_dir)

        try:
            image = Image.open(image_path).convert("RGB").resize((224, 224))
        except Exception as e:
            logger.warning("Image load failed, falling back to black image: %s (%s)", image_path, e)
            image = Image.new("RGB", (224, 224), color="black")

        text = str(row["hashtag"])
        label = int(row["label"])

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "sample_id": int(row["sample_id"]),
            "post_id": row["post_id"],
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
