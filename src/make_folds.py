import logging
import os
import sys

from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import DEFAULT_CSV_PATH, DEFAULT_IMAGES_DIR, PROJECT_ROOT, load_dataframe  # noqa: E402

logger = logging.getLogger(__name__)

SEED = 42
N_SPLITS = 5
DEFAULT_OUT_PATH = os.path.join(PROJECT_ROOT, "data", "folds.csv")


def make_folds(csv_path=DEFAULT_CSV_PATH, images_dir=DEFAULT_IMAGES_DIR, out_path=DEFAULT_OUT_PATH):
    df = load_dataframe(csv_path, images_dir).reset_index(drop=True)

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for fold_idx, (_, test_idx) in enumerate(cv.split(X=df, y=df["label"], groups=df["post_id"])):
        df.loc[test_idx, "fold"] = fold_idx

    assert (df["fold"] >= 0).all(), "Some rows were not assigned to any fold"

    post_fold_counts = df.groupby("post_id")["fold"].nunique()
    bad_posts = post_fold_counts[post_fold_counts != 1]
    assert bad_posts.empty, f"A post_id spans multiple folds: {bad_posts.index.tolist()}"

    folds_df = df[["sample_id", "post_id", "label", "fold"]].copy()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    folds_df.to_csv(out_path, index=False)
    logger.info("folds.csv saved: %s (%d rows)", out_path, len(folds_df))

    validate_folds(folds_df)
    return folds_df


def validate_folds(df):
    print(f"\n{'=' * 70}\nFold validation\n{'=' * 70}")
    overall_ok = True

    for fold in range(N_SPLITS):
        train_df = df[df.fold != fold]
        test_df = df[df.fold == fold]
        train_posts = set(train_df["post_id"])
        test_posts = set(test_df["post_id"])
        overlap = train_posts & test_posts
        ok = len(overlap) == 0
        overall_ok = overall_ok and ok

        print(f"\nFold {fold}:")
        print(f"  train pairs = {len(train_df)}, test pairs = {len(test_df)}")
        print(f"  train unique posts = {len(train_posts)}, test unique posts = {len(test_posts)}")
        print(f"  post overlap = {len(overlap)} {'[OK]' if ok else '[FAIL]'}")
        print(f"  train label dist:\n{train_df['label'].value_counts().sort_index().to_string()}")
        print(f"  test label dist:\n{test_df['label'].value_counts().sort_index().to_string()}")

        assert ok, f"Fold {fold}: post_id overlap between train/test: {overlap}"

    print(f"\n{'=' * 70}")
    print("ALL FOLD VALIDATIONS PASSED" if overall_ok else "FOLD VALIDATION FAILED")
    print(f"{'=' * 70}\n")

    return overall_ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    make_folds()
