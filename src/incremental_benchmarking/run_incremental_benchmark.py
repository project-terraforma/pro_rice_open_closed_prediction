"""
Incremental LR Benchmark vs Single-Run Baseline
================================================
1. Split project_c_samples.csv -> 20% held-out test + 80% training pool
2. Divide training pool evenly into 5 batches
3. Incremental run: warm-start LR across batches 1-5, evaluate on test after each
4. Single run: train on all 5 batches at once, evaluate on same test set
5. Save all splits to data/project_c_samples/batches/
6. Write results table to src/incremental_benchmarking/incremental_results.md
"""
from __future__ import annotations

import ast
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "project_c_samples"
BATCHES_DIR = DATA_DIR / "batches"
RESULTS_DIR = Path(__file__).resolve().parent
SRC_V2 = ROOT / "src" / "models_v2"
sys.path.insert(0, str(SRC_V2))

from shared_featurizer import SharedPlaceFeaturizer  # noqa: E402

BATCHES_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_FRACTION = 0.20
N_BATCHES = 5
FEATURE_BUNDLE = "low_plus_medium"
N_EPOCHS_PER_BATCH = 20
LR_ALPHA = 0.001

# ── CSV parsing ─────────────────────────────────────────────────────────────
_PARSE_COLS = ["sources", "names", "categories", "websites",
               "socials", "emails", "phones", "brand", "addresses"]

# Columns that must be lists (never None) so len() calls in the featurizer work
_LIST_COLS = ["sources", "names", "websites", "socials", "emails", "phones", "addresses"]


def _parse_field(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if not isinstance(val, str):
        return val
    val = val.strip()
    if val in ("", "None", "null", "nan"):
        return None
    try:
        return ast.literal_eval(val)
    except Exception:
        return None


def parse_csv_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in _PARSE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_field)
    # Replace None with [] for list columns so len() never raises TypeError
    for col in _LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    return df


def load_data(path: Path) -> pd.DataFrame:
    return parse_csv_df(pd.read_csv(path))


# ── Splitting helpers ────────────────────────────────────────────────────────
def make_batches(pool: pd.DataFrame, n: int, seed: int) -> list[pd.DataFrame]:
    shuffled = pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    batch_size = math.ceil(len(shuffled) / n)
    return [
        shuffled.iloc[i * batch_size: min((i + 1) * batch_size, len(shuffled))].reset_index(drop=True)
        for i in range(n)
    ]


# ── Featurisation ────────────────────────────────────────────────────────────
def featurize(train_df: pd.DataFrame, eval_df: pd.DataFrame):
    feat = SharedPlaceFeaturizer(
        feature_bundle=FEATURE_BUNDLE,
        use_source_confidence=False,
        use_interactions=True,
        auto_fit_on_transform=False,
    )
    feat.fit(train_df, label_col="open")
    X_tr = feat.transform(train_df)
    X_ev = feat.transform(eval_df)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ev = scaler.transform(X_ev)
    return X_tr, X_ev


# ── Metrics ──────────────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred, y_prob, label: str, n_train: int) -> dict:
    return dict(
        label=label,
        n_train=n_train,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        pr_auc=average_precision_score(y_true, y_prob),
    )


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print("Incremental LR Benchmark")
    print("=" * 65)

    df = load_data(DATA_DIR / "project_c_samples.csv")
    print(f"\nLoaded {len(df):,} rows  |  open={df['open'].sum()}  closed={(df['open']==0).sum()}")

    train_pool, test_df = train_test_split(
        df, test_size=TEST_FRACTION, random_state=RANDOM_SEED, stratify=df["open"]
    )
    train_pool = train_pool.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Test set  : {len(test_df):,} rows ({len(test_df)/len(df):.1%})")
    print(f"Train pool: {len(train_pool):,} rows ({len(train_pool)/len(df):.1%})")

    batches = make_batches(train_pool, N_BATCHES, RANDOM_SEED)
    for i, b in enumerate(batches, 1):
        print(f"  Batch {i}: {len(b)} rows | open={b['open'].sum()} closed={(b['open']==0).sum()}")

    # Save splits
    test_df.to_csv(BATCHES_DIR / "test_set.csv", index=False)
    pd.concat(batches, ignore_index=True).to_csv(BATCHES_DIR / "all_batches_combined.csv", index=False)
    for i, b in enumerate(batches, 1):
        b.to_csv(BATCHES_DIR / f"batch_{i}.csv", index=False)
    print(f"\nSaved test set, 5 batches, all_batches_combined → {BATCHES_DIR}")

    # ── Incremental run ──────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("INCREMENTAL RUN (warm-start partial_fit, weights never reset)")
    print("─" * 65)

    classes = np.array([0, 1])
    clf = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=LR_ALPHA,
        max_iter=1, tol=None, random_state=RANDOM_SEED,
        warm_start=True, n_jobs=-1,
    )
    cumulative = pd.DataFrame()
    y_test = test_df["open"].values.astype(int)
    inc_results: list[dict] = []

    for bi, batch in enumerate(batches, 1):
        cumulative = pd.concat([cumulative, batch], ignore_index=True)
        X_tr, X_te = featurize(cumulative, test_df)
        y_cum = cumulative["open"].values.astype(int)
        sw = compute_sample_weight("balanced", y_cum)
        for _ in range(N_EPOCHS_PER_BATCH):
            clf.partial_fit(X_tr, y_cum, classes=classes, sample_weight=sw)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        m = get_metrics(y_test, y_pred, y_prob, f"Batch {bi}", len(cumulative))
        inc_results.append(m)
        print(
            f"  After batch {bi} ({len(cumulative):>4} rows cumul.):  "
            f"acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  pr_auc={m['pr_auc']:.4f}"
        )

    # ── Single run ───────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("SINGLE RUN (train on all 5 batches at once)")
    print("─" * 65)

    all_bat = pd.concat(batches, ignore_index=True)
    X_tr_s, X_te_s = featurize(all_bat, test_df)
    y_all = all_bat["open"].values.astype(int)
    sw_all = compute_sample_weight("balanced", y_all)
    clf_s = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=LR_ALPHA,
        max_iter=1, tol=None, random_state=RANDOM_SEED,
        warm_start=True, n_jobs=-1,
    )
    for _ in range(N_EPOCHS_PER_BATCH):
        clf_s.partial_fit(X_tr_s, y_all, classes=classes, sample_weight=sw_all)
    yp_s = clf_s.predict(X_te_s)
    ypr_s = clf_s.predict_proba(X_te_s)[:, 1]
    m_s = get_metrics(y_test, yp_s, ypr_s, "Single run", len(all_bat))
    print(f"  acc={m_s['accuracy']:.4f}  f1={m_s['f1']:.4f}  pr_auc={m_s['pr_auc']:.4f}")

    delta = abs(inc_results[-1]["accuracy"] - m_s["accuracy"])
    within = delta <= 0.03
    print(f"\n  |Incremental final − Single run| = {delta:.4f}  "
          f"({'✅ within 3%' if within else '⚠️  outside 3%'})")

    # ── Write markdown ────────────────────────────────────────────────────
    training_labels = [
        "Batch 1 only",
        "Batches 1–2",
        "Batches 1–3",
        "Batches 1–4",
        "Batches 1–5",
    ]

    lines: list[str] = [
        "# Incremental LR Benchmark Results",
        "",
        "## Setup",
        "",
        f"- **Dataset**: `data/project_c_samples/project_c_samples.csv` ({len(df):,} rows)",
        f"- **Test set**: 20% stratified hold-out ({len(test_df):,} rows, fixed throughout)",
        f"- **Training pool**: 80% ({len(train_pool):,} rows) split evenly into {N_BATCHES} batches",
        f"- **Model**: Logistic Regression via `SGDClassifier` (log-loss, L2, α={LR_ALPHA}, {N_EPOCHS_PER_BATCH} epochs/batch)",
        "- **Warm-start**: `partial_fit` — model weights are **never reset** between batches",
        "- **Class imbalance**: balanced sample weights recomputed on the cumulative training set each batch",
        f"- **Feature bundle**: `{FEATURE_BUNDLE}`",
        "",
        "## Batch Sizes",
        "",
        "| Batch | Rows | Open | Closed |",
        "|-------|-----:|-----:|-------:|",
    ]
    for i, b in enumerate(batches, 1):
        lines.append(f"| {i} | {len(b)} | {b['open'].sum()} | {(b['open']==0).sum()} |")
    lines += [
        f"| **Test set** | **{len(test_df)}** | **{test_df['open'].sum()}** | **{(test_df['open']==0).sum()}** |",
        "",
        "## Results",
        "",
        "> Evaluation is always on the **same frozen test set** (20% hold-out).",
        "",
        "| Run | Training Data | Rows trained | Accuracy | Precision (open) | Recall (open) | F1 (open) | PR-AUC (open) |",
        "|-----|---------------|-------------:|---------:|-----------------:|--------------:|----------:|--------------:|",
    ]
    for r, tl in zip(inc_results, training_labels):
        lines.append(
            f"| Incremental — Batch {r['label'].split()[-1]} | {tl} | {r['n_train']} "
            f"| {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} "
            f"| {r['f1']:.4f} | {r['pr_auc']:.4f} |"
        )
    lines.append(
        f"| **Single run (all at once)** | Batches 1–5 (combined) | {len(all_bat)} "
        f"| **{m_s['accuracy']:.4f}** | {m_s['precision']:.4f} | {m_s['recall']:.4f} "
        f"| {m_s['f1']:.4f} | {m_s['pr_auc']:.4f} |"
    )

    lines += [
        "",
        "## Key Comparison",
        "",
        "| Metric | Incremental (after batch 5) | Single run | Δ |",
        "|--------|----------------------------:|-----------:|--:|",
        f"| Accuracy  | {inc_results[-1]['accuracy']:.4f} | {m_s['accuracy']:.4f} | {delta:+.4f} |",
        f"| F1 (open) | {inc_results[-1]['f1']:.4f} | {m_s['f1']:.4f} | {abs(inc_results[-1]['f1']-m_s['f1']):+.4f} |",
        f"| PR-AUC    | {inc_results[-1]['pr_auc']:.4f} | {m_s['pr_auc']:.4f} | {abs(inc_results[-1]['pr_auc']-m_s['pr_auc']):+.4f} |",
        "",
        "### Conclusion",
        "",
    ]
    if within:
        lines.append(
            f"✅ **Theory confirmed**: the absolute accuracy gap is **{delta:.4f}** (≤ 3%). "
            "Warm-start incremental training on successive data releases achieves comparable "
            "performance to training on all data at once — with no full retrain required."
        )
    else:
        lines.append(
            f"⚠️ **Gap of {delta:.4f} exceeds 3%** — incremental warm-start does not fully match "
            "single-run performance on this dataset. Consider more epochs per batch, "
            "a larger cumulative replay window, or a smaller regularisation (higher C)."
        )

    lines += [
        "",
        "## File Artifacts",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `data/project_c_samples/batches/test_set.csv` | Fixed 20% held-out test set |",
        "| `data/project_c_samples/batches/batch_1.csv` … `batch_5.csv` | Individual training batches |",
        "| `data/project_c_samples/batches/all_batches_combined.csv` | All 5 batches combined (single-run input) |",
        "",
    ]

    md_path = RESULTS_DIR / "incremental_results.md"
    md_path.write_text("\n".join(lines))
    print(f"\nResults written → {md_path}")
    print("Done.")


if __name__ == "__main__":
    main()
