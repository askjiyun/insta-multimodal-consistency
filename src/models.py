import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

MODEL_REGISTRY = {
    "clip_sim": {
        "type": "similarity",
        "backbone_repo": "openai/clip-vit-base-patch32",
        "loader": "clip",
    },
    "koclip_sim": {
        "type": "similarity",
        "backbone_repo": "Bingsu/clip-vit-large-patch14-ko",
        "loader": "auto",
    },
    "clip_frozen": {
        "type": "frozen",
        "backbone_repo": "openai/clip-vit-base-patch32",
        "loader": "clip",
        "freeze_backbone": True,
    },
    "koclip_frozen": {
        "type": "frozen",
        "backbone_repo": "Bingsu/clip-vit-large-patch14-ko",
        "loader": "auto",
        "freeze_backbone": True,
    },
    "clip_ft": {
        "type": "finetune",
        "backbone_repo": "openai/clip-vit-base-patch32",
        "loader": "clip",
        "freeze_backbone": False,
    },
    "koclip_ft": {
        "type": "finetune",
        "backbone_repo": "Bingsu/clip-vit-large-patch14-ko",
        "loader": "auto",
        "freeze_backbone": False,
    },
    "koclip_svm": {
        "type": "svm",
        "backbone_repo": "Bingsu/clip-vit-large-patch14-ko",
        "loader": "auto",
    },
    "koclip_xgb": {
        "type": "xgboost",
        "backbone_repo": "Bingsu/clip-vit-large-patch14-ko",
        "loader": "auto",
    },
}

TRAIN_PROTOCOL = {
    "seed": 42,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "max_epochs": 20,
    "patience": 3,
    "monitor": "qwk",
    "inner_val_size": 0.1,  # GroupShuffleSplit
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.1,
    "dropout": 0.3,
    "effective_batch_size": 32,
    "batch": {
        "frozen": {"batch_size": 32, "accum_steps": 1},
        "finetune": {"batch_size": 8, "accum_steps": 4},
    },
}

CLASSICAL_PROTOCOL = {
    "seed": 42,
    "inner_val_size": 0.1,
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
    },
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 20,
        "eval_metric": "mlogloss",
    },
}


class ClipClassifier(nn.Module):
    """image_embeds ⊕ text_embeds -> Linear(proj_dim*2->256)->ReLU->Dropout(0.3)->Linear(256->5)."""

    def __init__(self, backbone, freeze_backbone, num_classes=5, dropout=TRAIN_PROTOCOL["dropout"]):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        proj_dim = _get_projection_dim(self.backbone)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(
                    pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
                )
        else:
            outputs = self.backbone(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        combined = torch.cat((outputs.image_embeds, outputs.text_embeds), dim=1)
        return self.classifier(combined)


def _get_projection_dim(backbone):
    if hasattr(backbone, "config") and hasattr(backbone.config, "projection_dim"):
        return backbone.config.projection_dim
    if hasattr(backbone, "text_projection") and hasattr(backbone.text_projection, "out_features"):
        return backbone.text_projection.out_features
    raise ValueError("Could not find projection_dim in backbone")


def load_backbone_and_processor(repo, loader):
    if loader == "clip":
        model = CLIPModel.from_pretrained(repo)
        processor = CLIPProcessor.from_pretrained(repo)
    elif loader == "auto":
        model = AutoModel.from_pretrained(repo)
        processor = AutoProcessor.from_pretrained(repo)
    else:
        raise ValueError(f"Unknown loader: {loader}")
    return model, processor


FEATURE_ONLY_TYPES = ("similarity", "svm", "xgboost")


def build_model(model_key, num_classes=5):
    """model_key -> (model_or_backbone, processor)."""
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_key: {model_key} (registry: {list(MODEL_REGISTRY)})")

    cfg = MODEL_REGISTRY[model_key]
    backbone, processor = load_backbone_and_processor(cfg["backbone_repo"], cfg["loader"])

    if cfg["type"] in FEATURE_ONLY_TYPES:
        return backbone, processor

    model = ClipClassifier(backbone, freeze_backbone=cfg["freeze_backbone"], num_classes=num_classes)
    return model, processor
