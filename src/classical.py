import logging
import os
import sys

import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import CLASSICAL_PROTOCOL, MODEL_REGISTRY  # noqa: E402

logger = logging.getLogger(__name__)

SEED = CLASSICAL_PROTOCOL["seed"]


def _outer_split_masks(feat_df, folds_df, fold):
    fold_map = folds_df.set_index("sample_id")["fold"]
    row_fold = feat_df["sample_id"].map(fold_map)
    assert not row_fold.isna().any(), "feat_df has sample_id(s) not present in folds.csv"

    row_fold = row_fold.to_numpy()
    train_mask = row_fold != fold
    test_mask = row_fold == fold
    return train_mask, test_mask


def run_classical_fold(model_key, fold, X, feat_df, folds_df, output_dir=None, seed=SEED):
    cfg = MODEL_REGISTRY[model_key]
    if cfg["type"] not in ("svm", "xgboost"):
        raise ValueError(f"run_classical_fold only supports svm/xgboost types: {model_key} ({cfg['type']})")

    train_mask, test_mask = _outer_split_masks(feat_df, folds_df, fold)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train = feat_df.loc[train_mask, "label"].to_numpy()
    y_test = feat_df.loc[test_mask, "label"].to_numpy()
    test_sample_ids = feat_df.loc[test_mask, "sample_id"].tolist()
    test_post_ids = feat_df.loc[test_mask, "post_id"].tolist()

    if cfg["type"] == "svm":
        y_pred, artifact = _fit_predict_svm(X_train, y_train, X_test, seed)
    else:
        train_feat_df = feat_df.loc[train_mask].reset_index(drop=True)
        y_pred, artifact = _fit_predict_xgb(X_train, y_train, X_test, train_feat_df, seed)

    checkpoint_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "model.joblib")
        joblib.dump(artifact, checkpoint_path)

    return {
        "model_key": model_key,
        "fold": fold,
        "y_true": y_test.tolist(),
        "y_pred": list(y_pred),
        "sample_ids": test_sample_ids,
        "post_ids": test_post_ids,
        "checkpoint_path": checkpoint_path,
    }


def _fit_predict_svm(X_train, y_train, X_test, seed):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    params = dict(CLASSICAL_PROTOCOL["svm"])
    clf = SVC(random_state=seed, **params)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    return y_pred, {"scaler": scaler, "model": clf}


def _fit_predict_xgb(X_train, y_train, X_test, train_feat_df, seed):
    from xgboost import XGBClassifier

    gss = GroupShuffleSplit(n_splits=1, test_size=CLASSICAL_PROTOCOL["inner_val_size"], random_state=seed)
    inner_train_idx, inner_val_idx = next(gss.split(train_feat_df, groups=train_feat_df["post_id"]))

    overlap = set(train_feat_df.iloc[inner_train_idx]["post_id"]) & set(train_feat_df.iloc[inner_val_idx]["post_id"])
    assert not overlap, f"post_id overlap between inner train/val: {overlap}"

    X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
    y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]

    sample_weight = compute_sample_weight("balanced", y_inner_train)

    params = dict(CLASSICAL_PROTOCOL["xgboost"])
    early_stopping_rounds = params.pop("early_stopping_rounds")
    clf = XGBClassifier(
        objective="multi:softmax",
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )
    clf.fit(
        X_inner_train, y_inner_train,
        sample_weight=sample_weight,
        eval_set=[(X_inner_val, y_inner_val)],
        verbose=False,
    )
    logger.info("[xgboost] best_iteration=%s (n_estimators upper bound=%s)", clf.best_iteration, params["n_estimators"])
    y_pred = clf.predict(X_test)

    return y_pred, {"model": clf}
