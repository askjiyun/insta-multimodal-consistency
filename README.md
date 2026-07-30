# insta-multimodal-consistency

Reproducible experiments for **Korean image–hashtag semantic consistency** using **CLIP / KoCLIP**.

This repository accompanies the paper:

> **Predicting Semantic Consistency between Images and Korean Hashtags on Instagram**
> Jiyoon Oh & Jangmin Oh — School of AI Convergence, Sungshin Women's University

---

## Table of Contents

- [Description](#description)
- [Repository Structure](#repository-structure)
- [Dataset Information](#dataset-information)
- [Code Information](#code-information)
- [Requirements](#requirements)
- [Usage Instructions](#usage-instructions)
- [Methodology](#methodology)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Description

On image-centric platforms such as Instagram, user-generated hashtags often express subjective feelings, slang, or abstract concepts rather than literal descriptions of the image. This creates a semantic gap that general-purpose multimodal models (e.g., CLIP) struggle to capture, especially for Korean-language content.

This repository frames the problem as a **five-class semantic consistency prediction task** (scores 1–5) between an image and a hashtag, and compares three progressive training strategies across both CLIP and KoCLIP backbones:

1. Similarity-based baselines
2. Frozen-backbone classifiers
3. End-to-end fine-tuning

All annotated data, cross-validation splits, evaluation outputs, and code needed to reproduce the reported results are provided (images are hosted separately; see [Image Files](#image-files)).

---

## Repository Structure

```text
insta-multimodal-consistency/
│
├── README.md
├── pyproject.toml            # Project + dependency spec (managed with uv)
├── uv.lock                   # Locked dependency versions
├── run_cv.py                 # Entry point: cross-validation training & evaluation
│
├── annotation/               # LLM annotation pipeline
│   ├── gpt_labeling.py       # GPT-4o-mini labeling script (temperature=0.0)
│   └── labels_data.csv       # Raw annotation output (scores + Korean rationales)
│
├── data/
│   ├── dataset.csv           # Final training dataset (image–hashtag pairs, label 1–5)
│   ├── dataset_with_en.csv   # Same as dataset.csv + English hashtag translations
│   ├── folds.csv             # 5-fold cross-validation split (see note below)
│   ├── categorical_codebook.csv     # Machine-readable variable/label codebook
│   └── English-language-codebook.md # Full English data dictionary
│
├── images/                   # Instagram images (NOT in repo — see Image Files)
│
├── src/
│   ├── make_folds.py         # Builds folds.csv (StratifiedGroupKFold by post_id)
│   ├── datasets.py           # Dataset / dataloader definitions
│   ├── data_utils.py         # Data loading & preprocessing helpers
│   ├── features.py           # Embedding / feature extraction
│   ├── models.py             # CLIP / KoCLIP model + classifier head
│   ├── similarity.py         # Similarity-based baselines (+ threshold search)
│   ├── classical.py          # Classical ML baselines (SVM, XGBoost on embeddings)
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation driver
│   └── metrics.py            # Accuracy, macro/weighted-F1, QWK, MAE, adjacent acc, ...
│
└── outputs/                  # Experiment results
    ├── fold_metrics.csv      # Per-fold metrics, aggregated across all models
    ├── summary_metrics.csv   # Summary metrics table (paper-facing)
    ├── oof_predictions.csv   # Out-of-fold predictions (basis for significance tests)
    └── <model>/              # One folder per configuration:
        │                     #   clip_sim, clip_frozen, clip_ft,
        │                     #   koclip_sim, koclip_frozen, koclip_ft,
        │                     #   koclip_svm, koclip_xgb
        ├── confusion.csv
        ├── env_info.json     # Environment record (versions, CUDA, GPU)
        ├── fold_metrics.csv
        └── oof.csv
```

Model weights (`.pt` / `.pth`) are not committed; all CSV/JSON outputs are, so the reported numbers can be inspected without re-running training.

---

## Dataset Information

### Overview

The dataset contains **12,011 image–hashtag pairs** drawn from **1,673 public Korean Instagram posts**, covering **6,639 unique hashtags** across **8 thematic domains** (Travel, Food, Beauty/Shopping, Fashion, Season/Nature, Health/Diet, Tech/Product, Entertainment), each represented by 4 seed hashtags (32 in total). Each pair is annotated with a semantic consistency score from 1 (very low) to 5 (very high). Per-domain post/pair/tag counts and the full domain–keyword mapping are given in `data/English-language-codebook.md`.

### Data files and schema

**`data/dataset.csv`** — the dataset used for training and evaluation:

| Column | Type | Description |
| --- | --- | --- |
| `sample_id` | integer | Unique id per (image, hashtag) pair (1–12011). Join key to `folds.csv`. |
| `post_id` | string | Image filename in `images/` (e.g., `ai_0001.png`). Prefix = collection domain. One image maps to several hashtags. |
| `hashtag` | string | A single Korean/English hashtag (usually prefixed with `#`). |
| `label` | integer | Semantic consistency score, **1–5** (human-readable scale). |

**`data/dataset_with_en.csv`** — identical to `dataset.csv` with one extra column `hashtag_en` (English translation of the hashtag). Provided for reader convenience only; **models were trained on the original Korean `hashtag` column.**

**`data/folds.csv`** — the exact 5-fold split used in the paper, generated with `StratifiedGroupKFold` grouped by `post_id` (so no image appears in more than one fold).

> ⚠️ **Note on label indexing.** `folds.csv` stores `label` on a **0–4** scale (= `dataset.csv` label − 1). This is the 0-indexed form consumed directly by the training code (PyTorch cross-entropy expects class indices starting at 0). The offset is intentional — it is **not** an inconsistency. When joining, take `fold` from `folds.csv` and `label` from `dataset.csv` (or add 1 to the `folds.csv` label) to work on the 1–5 scale.

Full variable definitions are in `data/English-language-codebook.md` and `data/categorical_codebook.csv`.

### Consistency score codebook

| Score | Label | Meaning |
| --- | --- | --- |
| 1 | Very Low | 거의 관련 없음 — no visual/semantic support |
| 2 | Low | 낮은 관련성 — weak or indirect relation |
| 3 | Medium | 중간 관련성 — partially related, not the main subject |
| 4 | High | 높은 관련성 — clearly relevant to a major element |
| 5 | Very High | 매우 높은 관련성 — directly describes the dominant content |

Labels were produced by an LLM annotation pipeline (see [Code Information](#code-information)) and validated on a random subset of **500 pairs independently annotated by four human annotators**: inter-annotator ordinal Krippendorff's α = 0.72 (95% CI 0.69–0.75), mean GPT–human quadratic-weighted κ = **0.68 ± 0.03**, exact agreement 49.6%, adjacent agreement 89.8%. See the paper's *Reliability of GPT-4o mini Annotations* section for the full analysis.

### Image Files

Due to file size and copyright/privacy considerations, the Instagram images are **not** included in this repository. The image archive is hosted separately and is **available upon request for research purposes**: please open a GitHub issue on this repository or contact the corresponding author (jangmin.oh@sungshin.ac.kr) to receive a download link.

After downloading, extract so that files are accessible as `images/<post_id>` (e.g., `images/ai_0001.png`). Filenames correspond exactly to the `post_id` column in `data/dataset.csv`. Images are provided for research reproducibility only; copyright remains with the original posters.

---

## Code Information

### Annotation materials (referenced in the paper)

The complete annotation materials are contained in [`annotation/gpt_labeling.py`](annotation/gpt_labeling.py):

- **Full prompt text & scoring rubric** — the `ANNOTATION_GUIDELINE` constant (the five score definitions plus the eight annotation rules);
- **Boundary-case rules** — the *Special cases* block of the same guideline (generic tags, place tags, emotion/mood tags, brand/promotional tags);
- **Output schema** — the `HashtagEval` / `AnalysisResult` Pydantic models enforcing typed structured output (integer score in 1–5 + Korean visual evidence + rationale);
- **Model configuration** — the `CONFIG` constant (`gpt-4o-mini`, `temperature=0.0`, prompt version, seed) and the retry/backoff logic.

The raw per-pair annotation output, including the model's Korean rationales, is in `annotation/labels_data.csv`.

### Source files

| Path | Role |
| --- | --- |
| `annotation/gpt_labeling.py` | Annotates each (image, hashtags) post with GPT-4o-mini at `temperature=0.0` using structured (Pydantic) output. Emits per-pair scores plus Korean `visual_evidence` and `reason`. Resumable, with retry/backoff. |
| `annotation/labels_data.csv` | The raw annotation output. `data/dataset.csv` is derived from this file. |
| `src/make_folds.py` | Builds `data/folds.csv` from `data/dataset.csv` using `StratifiedGroupKFold` grouped by `post_id`. |
| `src/similarity.py` | Similarity-based baselines, including the threshold-optimization search. |
| `src/classical.py` | Classical ML baselines (SVM, XGBoost) on frozen CLIP/KoCLIP embeddings. |
| `src/train.py`, `src/evaluate.py` | Training loop and evaluation driver. |
| `src/models.py`, `src/features.py`, `src/datasets.py`, `src/data_utils.py`, `src/metrics.py` | Model/head definitions, feature extraction, data pipeline, and metric computation. |
| `run_cv.py` | Runs a single model (`--model <key>`, required) through 5-fold CV and writes results to `outputs/<key>/`. |

Backbones: **CLIP** (`openai/clip-vit-base-patch32`) and **KoCLIP** (`Bingsu/clip-vit-large-patch14-ko`). Configurations reported: similarity-based baselines, frozen-backbone classifiers, and end-to-end fine-tuning for each backbone, plus classical ML baselines (SVM, XGBoost) on frozen embeddings — matching the `outputs/` folders (`clip_sim`, `clip_frozen`, `clip_ft`, `koclip_sim`, `koclip_frozen`, `koclip_ft`, `koclip_svm`, `koclip_xgb`).

---

## Requirements

**Python 3.11** with a CUDA-capable **GPU**. Dependencies are managed with [uv](https://github.com/astral-sh/uv) via `pyproject.toml` + `uv.lock`:

```bash
uv sync
```

This recreates the exact locked environment for training and evaluation. Note that the lock file pins CUDA (Linux GPU) PyTorch wheels, so `uv sync` is intended for the GPU machine, not macOS.

The optional annotation step (`annotation/gpt_labeling.py`; see [Usage Instructions](#usage-instructions)) additionally requires `openai` and `pydantic`, which are **not** part of the locked environment — install them separately (e.g., `uv pip install openai`; `pydantic` is installed as its dependency).

### Hardware

All experiments were run on an **NVIDIA TITAN RTX (24 GB VRAM)**. Classification-based models were trained for up to 20 epochs with an effective batch size of 32.

---

## Usage Instructions

Labels and cross-validation folds are already provided in `data/`, so reviewers
can go straight to training/evaluation without re-running annotation.

**1. Generate cross-validation folds** from `data/dataset.csv`:

```bash
python src/make_folds.py
```

This writes `data/folds.csv`. The committed `folds.csv` already reflects the split used in the paper, so this step is only needed to regenerate it.

**2. Train and evaluate.** `run_cv.py` runs a single model through 5-fold CV; `--model` is required:

```bash
python run_cv.py --model koclip_ft
```

To reproduce every `outputs/<model>/` folder, run once per model key:

```bash
for m in clip_sim clip_frozen clip_ft koclip_sim koclip_frozen koclip_ft koclip_svm koclip_xgb; do
  python run_cv.py --model "$m"
done
```

Results (per-fold and summary metrics, confusion matrices, out-of-fold predictions, environment info) are written to `outputs/`. Committed outputs are overwritten on re-run. Run `python run_cv.py --help` for the full list of options (e.g. `--max-epochs`, `--subset-posts` for a smoke test).

`annotation/gpt_labeling.py` is included for reference — it is the script that produced `annotation/labels_data.csv`, from which `data/dataset.csv` was derived. It is not intended to be re-run directly from this repository (it expects raw crawl inputs that aren't part of this release) and is documented here only for transparency about how the labels were generated.

---

## Methodology

1. **Data collection** — Public Korean Instagram posts across 8 domains (32 seed hashtags), with per-post image and hashtags.
2. **Automated labeling** — GPT-4o-mini structured output assigns 1–5 consistency scores under a fixed annotation guideline (`temperature=0.0` for determinism). Labels were validated against four independent human annotators on 500 pairs (mean quadratic-weighted κ = 0.68 ± 0.03; human–human ordinal α = 0.72).
3. **Evaluation protocol** — 5-fold `StratifiedGroupKFold` grouped by `post_id` (no image leakage across folds). Metrics include accuracy, macro-F1, weighted-F1, and ordinal-aware measures (QWK, MAE, adjacent accuracy), with confusion-matrix and per-class analysis.
4. **Models** — CLIP (`openai/clip-vit-base-patch32`) and KoCLIP (`Bingsu/clip-vit-large-patch14-ko`) backbones under three strategies (similarity / frozen+classifier / end-to-end fine-tuning), plus classical ML baselines (SVM, XGBoost) on frozen embeddings. Trained with cross-entropy loss (class weighting), Adam, and early stopping.

See the accompanying paper for full details.

---

## Citation

If you use this repository, please cite:

```bibtex
@article{oh_koclip_consistency,
  author  = {Oh, Jiyoon and Oh, Jangmin},
  title   = {Predicting Semantic Consistency between Images and Korean Hashtags on Instagram},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {Under review},
  url     = {https://github.com/askjiyun/insta-multimodal-consistency}
}
```

---

## Acknowledgments

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [OpenAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (`openai/clip-vit-base-patch32`)
- [KoCLIP](https://huggingface.co/Bingsu/clip-vit-large-patch14-ko) (`Bingsu/clip-vit-large-patch14-ko`)
- OpenAI GPT-4o-mini (automated annotation)

---

## License

- **Code** is released under the **MIT License** (see [LICENSE](LICENSE)).
- **Data** created by the authors (`data/*.csv`, `annotation/labels_data.csv` — annotation labels, rationales, and metadata) is released under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**; please cite the paper when using it.
- **Images** are *not* covered by either license: copyright remains with the original Instagram posters. The separately distributed image archive (available upon request) is provided strictly for research reproducibility.
