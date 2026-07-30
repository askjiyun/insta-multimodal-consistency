import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import ClipHashtagDataset  # noqa: E402
from metrics import LABELS, compute_metrics  # noqa: E402
from models import MODEL_REGISTRY, TRAIN_PROTOCOL, build_model  # noqa: E402

logger = logging.getLogger(__name__)

SEED = TRAIN_PROTOCOL["seed"]
NUM_CLASSES = len(LABELS)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inner_train_val_split(outer_train_df, val_size, seed=SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss.split(outer_train_df, groups=outer_train_df["post_id"]))
    train_df = outer_train_df.iloc[train_idx].reset_index(drop=True)
    val_df = outer_train_df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_df["post_id"]) & set(val_df["post_id"])
    assert not overlap, f"post_id overlap between inner train/val: {overlap}"
    return train_df, val_df


def compute_fold_class_weights(labels, num_classes=NUM_CLASSES):
    labels = np.asarray(labels)
    present = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=labels)

    full = np.ones(num_classes, dtype=np.float64)
    for cls, w in zip(present, weights):
        full[int(cls)] = w
    return torch.tensor(full, dtype=torch.float)


def _get_trainable_state(model, freeze_backbone):
    if freeze_backbone:
        return {k: v.detach().cpu().clone() for k, v in model.classifier.state_dict().items()}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_trainable_state(model, state, freeze_backbone):
    if freeze_backbone:
        model.classifier.load_state_dict(state)
    else:
        model.load_state_dict(state)


def _run_inference(model, loader, device, collect_ids=False):
    model.eval()
    all_true, all_pred = [], []
    all_sample_ids, all_post_ids = [], []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(pixel_values, input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu()

            all_true.extend(labels.tolist())
            all_pred.extend(preds.tolist())

            if collect_ids:
                sample_ids = batch["sample_id"]
                all_sample_ids.extend(sample_ids.tolist() if torch.is_tensor(sample_ids) else list(sample_ids))
                all_post_ids.extend(list(batch["post_id"]))

    if collect_ids:
        return all_true, all_pred, all_sample_ids, all_post_ids
    return all_true, all_pred


def train_fold(
    model_key, fold, dataset_df, folds_df,
    output_dir=None, max_epochs=None, images_dir=None, cleanup_checkpoint=False,
):
    cfg = MODEL_REGISTRY[model_key]
    if cfg["type"] not in ("frozen", "finetune"):
        raise ValueError(f"train_fold only supports frozen/finetune types: {model_key} ({cfg['type']})")

    set_seed(SEED)
    device = get_device()

    max_epochs = max_epochs if max_epochs is not None else TRAIN_PROTOCOL["max_epochs"]
    patience = TRAIN_PROTOCOL["patience"]
    lr = TRAIN_PROTOCOL["lr"]
    weight_decay = TRAIN_PROTOCOL["weight_decay"]
    batch_cfg = TRAIN_PROTOCOL["batch"][cfg["type"]]
    batch_size = batch_cfg["batch_size"]
    accum_steps = batch_cfg["accum_steps"]

    merged = dataset_df.merge(folds_df[["sample_id", "fold"]], on="sample_id", how="inner")
    outer_train_df = merged[merged["fold"] != fold].reset_index(drop=True)
    outer_test_df = merged[merged["fold"] == fold].reset_index(drop=True)

    inner_train_df, inner_val_df = inner_train_val_split(outer_train_df, TRAIN_PROTOCOL["inner_val_size"], SEED)

    model, processor = build_model(model_key, num_classes=NUM_CLASSES)
    model.to(device)

    ds_kwargs = {"images_dir": images_dir} if images_dir else {}
    train_ds = ClipHashtagDataset(inner_train_df, processor, **ds_kwargs)
    val_ds = ClipHashtagDataset(inner_val_df, processor, **ds_kwargs)
    test_ds = ClipHashtagDataset(outer_test_df, processor, **ds_kwargs)

    gen = torch.Generator()
    gen.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    class_weights = compute_fold_class_weights(inner_train_df["label"].values).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=TRAIN_PROTOCOL["scheduler_step_size"], gamma=TRAIN_PROTOCOL["scheduler_gamma"]
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_qwk = -float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        n_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"[{model_key}][fold {fold}] epoch {epoch + 1}/{max_epochs}", leave=False)

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            is_flush_step = ((step + 1) % accum_steps == 0) or (step + 1 == n_batches)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(pixel_values, input_ids, attention_mask)
                    loss = criterion(outputs, labels) / accum_steps
                scaler.scale(loss).backward()
                if is_flush_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(pixel_values, input_ids, attention_mask)
                loss = criterion(outputs, labels) / accum_steps
                loss.backward()
                if is_flush_step:
                    optimizer.step()
                    optimizer.zero_grad()

            batch_loss = loss.item() * accum_steps
            running_loss += batch_loss
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        scheduler.step()

        val_true, val_pred = _run_inference(model, val_loader, device)
        val_metrics = compute_metrics(val_true, val_pred, labels=LABELS)
        avg_train_loss = running_loss / n_batches

        logger.info(
            "[%s][fold %d] epoch %d/%d train_loss=%.4f val_qwk=%.4f val_macro_f1=%.4f val_acc=%.4f",
            model_key, fold, epoch + 1, max_epochs, avg_train_loss,
            val_metrics["qwk"], val_metrics["macro_f1"], val_metrics["accuracy"],
        )
        history.append({
            "epoch": epoch + 1, "train_loss": avg_train_loss,
            "val_qwk": val_metrics["qwk"], "val_macro_f1": val_metrics["macro_f1"],
        })

        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            best_state = _get_trainable_state(model, cfg["freeze_backbone"])
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    "[%s][fold %d] early stopping @ epoch %d (best inner-val qwk=%.4f)",
                    model_key, fold, epoch + 1, best_qwk,
                )
                break

    assert best_state is not None, "Training did not complete a single epoch"
    _load_trainable_state(model, best_state, cfg["freeze_backbone"])

    checkpoint_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "best.pth")
        torch.save(best_state, checkpoint_path)

    test_true, test_pred, test_sample_ids, test_post_ids = _run_inference(
        model, test_loader, device, collect_ids=True
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if cleanup_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("[%s][fold %d] checkpoint deleted: %s", model_key, fold, checkpoint_path)
        checkpoint_path = None

    return {
        "model_key": model_key,
        "fold": fold,
        "y_true": test_true,
        "y_pred": test_pred,
        "sample_ids": test_sample_ids,
        "post_ids": test_post_ids,
        "best_inner_val_qwk": best_qwk,
        "n_epochs_run": len(history),
        "checkpoint_path": checkpoint_path,
        "history": history,
    }
