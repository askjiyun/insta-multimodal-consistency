import glob
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import FOLD_METRIC_COLUMNS, LABELS, fold_metrics_row  # noqa: E402

logger = logging.getLogger(__name__)

OOF_COLUMNS = ["sample_id", "post_id", "true_label", "pred_label", "fold", "model"]
METRIC_COLS = ["accuracy", "macro_f1", "weighted_f1", "mae", "qwk", "adj_acc"]


def model_output_dir(outputs_dir, model_key):
    return os.path.join(outputs_dir, model_key)


def fold_checkpoint_dir(outputs_dir, model_key, fold):
    return os.path.join(model_output_dir(outputs_dir, model_key), f"fold_{fold}")


def save_model_results(model_key, fold_results, outputs_dir, labels=LABELS):
    metric_rows = [
        fold_metrics_row(model_key, r["fold"], r["y_true"], r["y_pred"], labels=labels)
        for r in fold_results
    ]
    metrics_df = pd.DataFrame(metric_rows, columns=FOLD_METRIC_COLUMNS)

    oof_parts = []
    for r in fold_results:
        n = len(r["y_true"])
        oof_parts.append(pd.DataFrame({
            "sample_id": r["sample_ids"],
            "post_id": r["post_ids"],
            "true_label": r["y_true"],
            "pred_label": r["y_pred"],
            "fold": [r["fold"]] * n,
            "model": [model_key] * n,
        }))
    oof_df = pd.concat(oof_parts, ignore_index=True)[OOF_COLUMNS]

    dup = oof_df["sample_id"].duplicated()
    assert not dup.any(), (
        f"Model {model_key}: the same sample_id appears in OOF for multiple folds: "
        f"{oof_df.loc[dup, 'sample_id'].tolist()}"
    )

    out_dir = model_output_dir(outputs_dir, model_key)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "fold_metrics.csv")
    oof_path = os.path.join(out_dir, "oof.csv")
    metrics_df.to_csv(metrics_path, index=False)
    oof_df.to_csv(oof_path, index=False)
    logger.info("[%s] saved: %s (5 rows), %s (%d rows)", model_key, metrics_path, oof_path, len(oof_df))

    save_confusion_matrix(oof_df, model_key, out_dir, labels=labels)

    return metrics_df, oof_df


def save_confusion_matrix(oof_df, model_key, out_dir, labels=LABELS):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(oof_df["true_label"], oof_df["pred_label"], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    path = os.path.join(out_dir, "confusion.csv")
    cm_df.to_csv(path)
    logger.info("[%s] confusion matrix saved: %s", model_key, path)
    return path


def discover_completed_models(outputs_dir):
    pattern = os.path.join(outputs_dir, "*", "fold_metrics.csv")
    model_keys = []
    for path in sorted(glob.glob(pattern)):
        model_key = os.path.basename(os.path.dirname(path))
        model_keys.append(model_key)
    return model_keys


def aggregate_all(outputs_dir):
    model_keys = discover_completed_models(outputs_dir)
    if not model_keys:
        raise FileNotFoundError(f"No completed model results found under {outputs_dir}")

    fold_metrics_parts, oof_parts = [], []
    for model_key in model_keys:
        d = model_output_dir(outputs_dir, model_key)
        fold_metrics_parts.append(pd.read_csv(os.path.join(d, "fold_metrics.csv")))
        oof_parts.append(pd.read_csv(os.path.join(d, "oof.csv")))

    fold_metrics_df = pd.concat(fold_metrics_parts, ignore_index=True)[FOLD_METRIC_COLUMNS]
    oof_df = pd.concat(oof_parts, ignore_index=True)[OOF_COLUMNS]

    fold_metrics_path = os.path.join(outputs_dir, "fold_metrics.csv")
    oof_path = os.path.join(outputs_dir, "oof_predictions.csv")
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    oof_df.to_csv(oof_path, index=False)

    summary_df = compute_summary_metrics(fold_metrics_df)
    summary_path = os.path.join(outputs_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)

    logger.info(
        "Aggregation complete: models=%s fold_metrics=%d rows oof=%d rows -> %s",
        model_keys, len(fold_metrics_df), len(oof_df), outputs_dir,
    )
    return fold_metrics_df, oof_df, summary_df


def compute_summary_metrics(fold_metrics_df):
    agg = fold_metrics_df.groupby("model")[METRIC_COLS].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg["n_folds"] = fold_metrics_df.groupby("model").size()
    return agg.reset_index()


def validate_oof(oof_df, dataset_df):
    all_ok = True
    expected = set(dataset_df["sample_id"])
    for model_key, group in oof_df.groupby("model"):
        dup = group["sample_id"].duplicated()
        n_dup = int(dup.sum())
        actual = set(group["sample_id"])
        missing = expected - actual
        extra = actual - expected
        ok = n_dup == 0 and not missing and not extra
        all_ok = all_ok and ok
        logger.info(
            "[%s] OOF rows=%d dup=%d missing=%d extra=%d -> %s",
            model_key, len(group), n_dup, len(missing), len(extra), "OK" if ok else "FAIL",
        )
    return all_ok
