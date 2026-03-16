"""
Incremental Benchmark – All Models (LR, XGBoost, LightGBM, Random Forest)
==========================================================================
Compares two training strategies evaluated on the same frozen test set:

  **Incremental (warm-start)**
    Train a model on batch 1, evaluate.  Then *continue* training the same
    model on batch 2 (warm-start), evaluate again.  Repeat through batch 5.
    At step k the model has accumulated knowledge from batches 1 … k.

    • LR:       warm_start=True  (lbfgs resumes from previous coefficients)
    • RF:       warm_start=True  (adds trees each batch)
    • XGBoost:  xgb_model=prev   (continues boosting from prior booster)
    • LightGBM: init_model=prev  (continues boosting from prior model)

  **Single run**
    Train a FRESH model on the full training pool loaded from
    `data/project_c_samples/batches/all_batches_combined.csv` (2 740 rows).

Featurizer + StandardScaler are fitted **once** on the full training pool so
that the feature space is identical for every batch iteration and the single
run.  This avoids column-mismatch issues across batches while remaining fair
(featurizer never sees test labels).

Hyperparameters sourced from HPO artifacts (same as before).
Results written to: src/incremental_benchmarking/incremental_results.md
"""
from __future__ import annotations

import ast
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("xgboost required: pip install xgboost") from e

try:
    from lightgbm import LGBMClassifier
except ImportError as e:
    raise ImportError("lightgbm required: pip install lightgbm") from e

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data" / "project_c_samples"
BATCHES_DIR = DATA_DIR / "batches"
RESULTS_DIR = Path(__file__).resolve().parent
SRC_V2      = ROOT / "src" / "models_v2"
sys.path.insert(0, str(SRC_V2))

from shared_featurizer import SharedPlaceFeaturizer  # noqa: E402

BATCHES_DIR.mkdir(parents=True, exist_ok=True)

# ── Global config ─────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
TEST_FRACTION  = 0.20
N_BATCHES      = 5
FEATURE_BUNDLE = "low_plus_medium"

# ── HPO-sourced hyperparameters ───────────────────────────────────────────────
HPO_LR = {
    "C":            0.031066049133505004,
    "class_weight": {0: 4.5, 1: 1.0},
    "max_iter":     2000,
    "solver":       "lbfgs",
}

HPO_RF = {
    "class_weight":     "balanced",
    "max_depth":        8,
    "max_features":     "log2",
    "min_samples_leaf": 7,
    "min_samples_split": 6,
    "n_estimators":     375,
}

HPO_LGB = {
    "class_weight":    "balanced",
    "colsample_bytree": 0.9032350960341495,
    "learning_rate":   0.012106766445987193,
    "min_child_samples": 68,
    "n_estimators":    263,
    "num_leaves":      111,
    "reg_lambda":      0.5117430300313628,
    "subsample":       0.852665759648826,
}

HPO_XGB = {
    "n_estimators":    300,
    "learning_rate":   0.05,
    "max_depth":       6,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda":      1.0,
    "objective":       "binary:logistic",
    "eval_metric":     "logloss",
}

# Number of boosting rounds / trees to add per incremental batch
TREES_PER_BATCH_XGB = HPO_XGB["n_estimators"]   # 300 rounds per batch
TREES_PER_BATCH_LGB = HPO_LGB["n_estimators"]   # 263 rounds per batch
TREES_PER_BATCH_RF  = HPO_RF["n_estimators"]     # 375 trees per batch


# ── CSV parsing ───────────────────────────────────────────────────────────────
_PARSE_COLS = ["sources", "names", "categories", "websites",
               "socials", "emails", "phones", "brand", "addresses"]
_LIST_COLS  = ["sources", "names", "websites", "socials", "emails", "phones", "addresses"]


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
    for col in _LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    return df


def load_data(path: Path) -> pd.DataFrame:
    return parse_csv_df(pd.read_csv(path))


# ── Stratified batch splitting ────────────────────────────────────────────────
def make_batches(pool: pd.DataFrame, n: int, seed: int) -> list[pd.DataFrame]:
    """Split pool into n stratified batches preserving open/closed ratio."""
    open_rows   = pool[pool["open"] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
    closed_rows = pool[pool["open"] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)

    def _split(rows: pd.DataFrame) -> list[pd.DataFrame]:
        size = math.ceil(len(rows) / n)
        return [rows.iloc[i * size: min((i + 1) * size, len(rows))].reset_index(drop=True)
                for i in range(n)]

    open_splits   = _split(open_rows)
    closed_splits = _split(closed_rows)
    return [
        pd.concat([o, c], ignore_index=True)
        .sample(frac=1, random_state=seed + i)
        .reset_index(drop=True)
        for i, (o, c) in enumerate(zip(open_splits, closed_splits))
    ]


# ── Featurisation (fitted once, reused) ───────────────────────────────────────
def fit_featurizer_and_scaler(
    train_pool: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[SharedPlaceFeaturizer, StandardScaler, np.ndarray]:
    """
    Fit featurizer + scaler on the FULL training pool once.
    Returns (featurizer, scaler, X_test_scaled).
    """
    feat = SharedPlaceFeaturizer(
        feature_bundle=FEATURE_BUNDLE,
        use_source_confidence=False,
        use_interactions=True,
        auto_fit_on_transform=False,
    )
    feat.fit(train_pool, label_col="open")
    X_pool = feat.transform(train_pool)
    X_test = feat.transform(test_df)

    scaler = StandardScaler()
    scaler.fit(X_pool)                          # fit on full pool
    X_test_scaled = scaler.transform(X_test)

    print(f"  Featurizer fitted on {len(train_pool)} rows → {X_pool.shape[1]} features")
    return feat, scaler, X_test_scaled


def transform_batch(batch_df: pd.DataFrame, feat: SharedPlaceFeaturizer,
                    scaler: StandardScaler) -> np.ndarray:
    """Transform a single batch using the pre-fitted featurizer + scaler."""
    X = feat.transform(batch_df)
    return scaler.transform(X)


# ── Metrics ───────────────────────────────────────────────────────────────────
def get_metrics(y_true, y_pred, y_prob, label: str, model: str,
                n_train: int, total_rows_seen: int) -> dict:
    """
    y_prob is P(open=1).  All closed-class metrics use pos_label=0.
    """
    y_prob_closed = 1.0 - y_prob
    y_true_closed = (np.asarray(y_true) == 0).astype(int)
    return dict(
        model=model, label=label, n_train=n_train, total_rows_seen=total_rows_seen,
        accuracy=accuracy_score(y_true, y_pred),
        closed_precision=precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        closed_recall=recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        f1_closed=f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        pr_auc=average_precision_score(y_true_closed, y_prob_closed),
    )


# ── Evaluate helper ──────────────────────────────────────────────────────────
def _evaluate(model_tag: str, clf, X_test: np.ndarray, y_test: np.ndarray,
              run_label: str, n_train: int, total_rows_seen: int) -> dict:
    """Predict on frozen test set and compute metrics."""
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    m = get_metrics(y_test, y_pred, y_prob, run_label, model_tag,
                    n_train, total_rows_seen)
    print(f"    ⤷ acc={m['accuracy']:.4f}  closed_prec={m['closed_precision']:.4f}  "
          f"closed_rec={m['closed_recall']:.4f}  f1_closed={m['f1_closed']:.4f}  "
          f"pr_auc={m['pr_auc']:.4f}")
    return m


# ═════════════════════════════════════════════════════════════════════════════
# INCREMENTAL RUNNERS  (warm-start / continued training)
# ═════════════════════════════════════════════════════════════════════════════

def run_incremental_lr(batches, X_batches, y_batches, X_test, y_test):
    """LR: warm_start=True — lbfgs resumes from previous coefficients."""
    results = []
    clf = LogisticRegression(
        C=HPO_LR["C"], class_weight=HPO_LR["class_weight"],
        max_iter=HPO_LR["max_iter"], solver=HPO_LR["solver"],
        penalty="l2", random_state=RANDOM_SEED, n_jobs=-1,
        warm_start=True,
    )
    total_seen = 0
    for bi in range(len(batches)):
        total_seen += len(batches[bi])
        n_open   = int(y_batches[bi].sum())
        n_closed = int((y_batches[bi] == 0).sum())
        print(f"  [LR] Training on batch {bi+1}: {len(batches[bi])} rows "
              f"(open={n_open}, closed={n_closed})  |  total seen so far: {total_seen}")
        clf.fit(X_batches[bi], y_batches[bi])
        m = _evaluate("LR", clf, X_test, y_test, f"After batch {bi+1}",
                       len(batches[bi]), total_seen)
        results.append(m)
    return results


def run_incremental_xgb(batches, X_batches, y_batches, X_test, y_test):
    """XGBoost: xgb_model= continues boosting from prior booster."""
    results = []
    prev_model = None
    total_seen = 0
    for bi in range(len(batches)):
        total_seen += len(batches[bi])
        n_open   = int(y_batches[bi].sum())
        n_closed = int((y_batches[bi] == 0).sum())
        spw = n_open / n_closed if n_closed > 0 else 1.0
        print(f"  [XGBoost] Training on batch {bi+1}: {len(batches[bi])} rows "
              f"(open={n_open}, closed={n_closed}, spw={spw:.2f})  |  "
              f"total seen so far: {total_seen}")

        clf = XGBClassifier(
            n_estimators=TREES_PER_BATCH_XGB,
            learning_rate=HPO_XGB["learning_rate"],
            max_depth=HPO_XGB["max_depth"],
            subsample=HPO_XGB["subsample"],
            colsample_bytree=HPO_XGB["colsample_bytree"],
            min_child_weight=HPO_XGB["min_child_weight"],
            reg_lambda=HPO_XGB["reg_lambda"],
            objective=HPO_XGB["objective"],
            eval_metric=HPO_XGB["eval_metric"],
            scale_pos_weight=spw,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
        if prev_model is not None:
            clf.fit(X_batches[bi], y_batches[bi], xgb_model=prev_model.get_booster())
        else:
            clf.fit(X_batches[bi], y_batches[bi])
        prev_model = clf

        m = _evaluate("XGBoost", clf, X_test, y_test, f"After batch {bi+1}",
                       len(batches[bi]), total_seen)
        results.append(m)
    return results


def run_incremental_lgb(batches, X_batches, y_batches, X_test, y_test):
    """LightGBM: init_model= continues boosting from prior model."""
    results = []
    prev_model = None
    total_seen = 0
    for bi in range(len(batches)):
        total_seen += len(batches[bi])
        n_open   = int(y_batches[bi].sum())
        n_closed = int((y_batches[bi] == 0).sum())
        print(f"  [LightGBM] Training on batch {bi+1}: {len(batches[bi])} rows "
              f"(open={n_open}, closed={n_closed})  |  total seen so far: {total_seen}")

        clf = LGBMClassifier(
            n_estimators=TREES_PER_BATCH_LGB,
            learning_rate=HPO_LGB["learning_rate"],
            num_leaves=HPO_LGB["num_leaves"],
            min_child_samples=HPO_LGB["min_child_samples"],
            class_weight=HPO_LGB["class_weight"],
            colsample_bytree=HPO_LGB["colsample_bytree"],
            reg_lambda=HPO_LGB["reg_lambda"],
            subsample=HPO_LGB["subsample"],
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )
        if prev_model is not None:
            clf.fit(X_batches[bi], y_batches[bi], init_model=prev_model)
        else:
            clf.fit(X_batches[bi], y_batches[bi])
        prev_model = clf

        m = _evaluate("LightGBM", clf, X_test, y_test, f"After batch {bi+1}",
                       len(batches[bi]), total_seen)
        results.append(m)
    return results


def run_incremental_rf(batches, X_batches, y_batches, X_test, y_test):
    """RF: warm_start=True — each batch adds TREES_PER_BATCH_RF new trees."""
    results = []
    clf = RandomForestClassifier(
        max_depth=HPO_RF["max_depth"],
        max_features=HPO_RF["max_features"],
        min_samples_leaf=HPO_RF["min_samples_leaf"],
        min_samples_split=HPO_RF["min_samples_split"],
        class_weight=HPO_RF["class_weight"],
        n_estimators=TREES_PER_BATCH_RF,   # start with 375 trees
        random_state=RANDOM_SEED, n_jobs=-1,
        warm_start=True,
    )
    total_seen = 0
    for bi in range(len(batches)):
        total_seen += len(batches[bi])
        n_open   = int(y_batches[bi].sum())
        n_closed = int((y_batches[bi] == 0).sum())
        if bi > 0:
            # Add more trees for each subsequent batch
            clf.n_estimators += TREES_PER_BATCH_RF
        print(f"  [RF] Training on batch {bi+1}: {len(batches[bi])} rows "
              f"(open={n_open}, closed={n_closed})  |  "
              f"total trees={clf.n_estimators}  |  total seen so far: {total_seen}")
        clf.fit(X_batches[bi], y_batches[bi])
        m = _evaluate("RF", clf, X_test, y_test, f"After batch {bi+1}",
                       len(batches[bi]), total_seen)
        results.append(m)
    return results


# ── Single-run helper ─────────────────────────────────────────────────────────
def run_single(model_tag: str, clf, X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray, n_total: int) -> dict:
    """Train a fresh model on all training data at once."""
    n_open   = int(y_train.sum())
    n_closed = int((y_train == 0).sum())
    print(f"  [{model_tag}] Single run: {n_total} rows "
          f"(open={n_open}, closed={n_closed})")
    clf.fit(X_train, y_train)
    return _evaluate(model_tag, clf, X_test, y_test, "Single run",
                     n_total, n_total)


# ── Markdown helpers ──────────────────────────────────────────────────────────
def _fmt_row(cells: list[str], widths: list[int]) -> str:
    return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"


def _sep_row(widths: list[int], aligns: list[str]) -> str:
    parts = []
    for w, a in zip(widths, aligns):
        if a == "r":
            parts.append("-" * (w - 1) + ":")
        elif a == "c":
            parts.append(":" + "-" * (w - 2) + ":")
        else:
            parts.append("-" * w)
    return "| " + " | ".join(parts) + " |"


def _aligned_table(headers: list[str], rows: list[list[str]], aligns: list[str]) -> list[str]:
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    lines = [_fmt_row([h.ljust(w) for h, w in zip(headers, widths)], widths)]
    lines.append(_sep_row(widths, aligns))
    for row in rows:
        lines.append(_fmt_row(
            [str(c).ljust(w) if aligns[i] == "l" else str(c).rjust(w)
             for i, (c, w) in enumerate(zip(row, widths))],
            widths,
        ))
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("Incremental Benchmark — Warm-Start Training")
    print("  LR · XGBoost · LightGBM · Random Forest")
    print("=" * 70)

    # ── Load source data and split ────────────────────────────────────────────
    df = load_data(DATA_DIR / "project_c_samples.csv")
    print(f"\nLoaded {len(df):,} rows  |  open={df['open'].sum()}  closed={(df['open']==0).sum()}")

    train_pool, test_df = train_test_split(
        df, test_size=TEST_FRACTION, random_state=RANDOM_SEED, stratify=df["open"],
    )
    train_pool = train_pool.reset_index(drop=True)
    test_df    = test_df.reset_index(drop=True)
    y_test     = test_df["open"].values.astype(int)

    print(f"Test set  : {len(test_df):,} rows  |  open={test_df['open'].sum()}  "
          f"closed={(test_df['open']==0).sum()}")
    print(f"Train pool: {len(train_pool):,} rows")

    batches = make_batches(train_pool, N_BATCHES, RANDOM_SEED)
    print("\nStratified batch distribution:")
    for i, b in enumerate(batches, 1):
        print(f"  Batch {i}: {len(b):>4} rows | open={b['open'].sum():>4} | "
              f"closed={(b['open']==0).sum():>3}")

    # ── Save splits ───────────────────────────────────────────────────────────
    test_df.to_csv(BATCHES_DIR / "test_set.csv", index=False)
    for i, b in enumerate(batches, 1):
        b.to_csv(BATCHES_DIR / f"batch_{i}.csv", index=False)
    pd.concat(batches, ignore_index=True).to_csv(
        BATCHES_DIR / "all_batches_combined.csv", index=False,
    )
    print(f"\nSaved batches → {BATCHES_DIR}")

    # ── Load single-run training data independently ───────────────────────────
    single_train_df = load_data(BATCHES_DIR / "all_batches_combined.csv")
    print(f"\nSingle-run data: {len(single_train_df)} rows  |  "
          f"open={single_train_df['open'].sum()}  "
          f"closed={(single_train_df['open']==0).sum()}")

    # ── Fit featurizer + scaler ONCE on full training pool ────────────────────
    print("\n── Fitting shared featurizer + scaler on full training pool ──")
    feat, scaler, X_test = fit_featurizer_and_scaler(train_pool, test_df)

    # Pre-transform all batches and single-run data
    X_batches = [transform_batch(b, feat, scaler) for b in batches]
    y_batches = [b["open"].values.astype(int) for b in batches]

    X_single = transform_batch(single_train_df, feat, scaler)
    y_single = single_train_df["open"].values.astype(int)

    print(f"\n  Pre-transformed {N_BATCHES} batches + single-run data")
    for i, (Xb, yb) in enumerate(zip(X_batches, y_batches), 1):
        print(f"    Batch {i}: {Xb.shape}  open={yb.sum()}  closed={(yb==0).sum()}")
    print(f"    Single:  {X_single.shape}  open={y_single.sum()}  closed={(y_single==0).sum()}")

    # ── Run incremental (warm-start) for each model ───────────────────────────
    all_results: list[tuple[str, list[dict], dict]] = []

    for model_tag, inc_fn, single_make_fn in [
        ("LR",       run_incremental_lr,  lambda: LogisticRegression(
            C=HPO_LR["C"], class_weight=HPO_LR["class_weight"],
            max_iter=HPO_LR["max_iter"], solver=HPO_LR["solver"],
            penalty="l2", random_state=RANDOM_SEED, n_jobs=-1)),
        ("XGBoost",  run_incremental_xgb, lambda: XGBClassifier(
            **HPO_XGB,
            scale_pos_weight=int(y_single.sum()) / max(int((y_single==0).sum()), 1),
            random_state=RANDOM_SEED, n_jobs=-1)),
        ("LightGBM", run_incremental_lgb, lambda: LGBMClassifier(
            **HPO_LGB, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)),
        ("RF",       run_incremental_rf,  lambda: RandomForestClassifier(
            **HPO_RF, random_state=RANDOM_SEED, n_jobs=-1)),
    ]:
        print("\n" + "─" * 70)
        print(f"MODEL: {model_tag}")
        print("─" * 70)

        # Incremental (warm-start)
        print(f"\n  ── Incremental (warm-start) ──")
        inc_res = inc_fn(batches, X_batches, y_batches, X_test, y_test)

        # Single run
        print(f"\n  ── Single run ──")
        single_clf = single_make_fn()
        single_res = run_single(model_tag, single_clf, X_single, y_single,
                                X_test, y_test, len(single_train_df))

        all_results.append((model_tag, inc_res, single_res))

    # ── Build markdown ────────────────────────────────────────────────────────
    model_names = {
        "LR": "LR", "XGBoost": "XGBoost", "LightGBM": "LightGBM", "RF": "Random Forest",
    }

    lr_params_str  = (f"C={HPO_LR['C']:.4f}, class_weight={{0:{HPO_LR['class_weight'][0]}, "
                      f"1:{HPO_LR['class_weight'][1]}}}, max_iter={HPO_LR['max_iter']}, "
                      f"solver={HPO_LR['solver']}")
    rf_params_str  = (f"n_estimators={HPO_RF['n_estimators']}/batch, "
                      f"max_depth={HPO_RF['max_depth']}, "
                      f"max_features={HPO_RF['max_features']}, "
                      f"min_samples_leaf={HPO_RF['min_samples_leaf']}, "
                      f"min_samples_split={HPO_RF['min_samples_split']}, "
                      f"class_weight={HPO_RF['class_weight']}")
    lgb_params_str = (f"n_estimators={HPO_LGB['n_estimators']}/batch, "
                      f"lr={HPO_LGB['learning_rate']:.4f}, "
                      f"num_leaves={HPO_LGB['num_leaves']}, "
                      f"min_child_samples={HPO_LGB['min_child_samples']}, "
                      f"class_weight={HPO_LGB['class_weight']}")
    xgb_params_str = (f"n_estimators={HPO_XGB['n_estimators']}/batch, "
                      f"lr={HPO_XGB['learning_rate']}, "
                      f"max_depth={HPO_XGB['max_depth']}, "
                      f"scale_pos_weight=class-ratio")

    lines: list[str] = [
        "# Incremental Benchmark Results — All Models (Warm-Start)",
        "",
        "## Setup",
        "",
        f"- **Dataset**: `data/project_c_samples/project_c_samples.csv` ({len(df):,} rows)",
        f"- **Test set**: 20% stratified hold-out ({len(test_df):,} rows, fixed throughout)",
        f"- **Training pool**: 80% ({len(train_pool):,} rows) split into {N_BATCHES} "
        f"**stratified** batches (equal open/closed ratio per batch)",
        f"- **Feature bundle**: `{FEATURE_BUNDLE}`",
        "- **Featurizer + scaler**: fitted once on full training pool, frozen for all runs",
        "",
        "### Hyperparameters (from HPO artifacts)",
        "",
        "| Model         | Source | Key params |",
        "|:--------------|:-------|:-----------|",
        f"| LR            | `hpo_optuna_lr_micro_pass1` | {lr_params_str} |",
        f"| XGBoost       | `xgboost_model_v2.py` defaults | {xgb_params_str} |",
        f"| LightGBM      | `lightgbm_hpo/hpo_selected_trials.csv` | {lgb_params_str} |",
        f"| Random Forest | `hpo_optuna_rf_micro_pass1` | {rf_params_str} |",
        "",
        "### Training strategy",
        "",
        "| Run type | Description |",
        "|:---------|:------------|",
        "| Incremental — After batch *k* | **Warm-start**: model trained on batch 1, then continued on batch 2, … through batch *k*. Accumulated knowledge from all prior batches. |",
        "| Single run (all at once) | Train a **fresh** model on `all_batches_combined.csv` (2,740 rows) |",
        "",
        "#### Warm-start mechanism per model",
        "",
        "| Model | Mechanism | What happens each batch |",
        "|:------|:----------|:------------------------|",
        "| LR | `warm_start=True` | lbfgs resumes from previous coefficients |",
        f"| XGBoost | `xgb_model=prev_booster` | adds {TREES_PER_BATCH_XGB} new boosting rounds on top of prior model |",
        f"| LightGBM | `init_model=prev_model` | adds {TREES_PER_BATCH_LGB} new boosting rounds on top of prior model |",
        f"| Random Forest | `warm_start=True` | adds {TREES_PER_BATCH_RF} new trees each batch |",
        "",
        "> The featurizer and scaler are fitted **once** on the full training pool",
        "> to ensure a consistent feature space across all batch iterations.",
        "> No test labels are ever seen during featurizer fitting.",
        "",
        "---",
        "",
        "## Batch Sizes (Stratified)",
        "",
    ]

    b_headers = ["Batch", "Rows", "Open", "Closed"]
    b_aligns  = ["l",     "r",    "r",    "r"]
    b_rows = [[str(i), str(len(b)), str(b['open'].sum()), str((b['open']==0).sum())]
              for i, b in enumerate(batches, 1)]
    b_rows.append(["**Test set**",
                   f"**{len(test_df)}**",
                   f"**{test_df['open'].sum()}**",
                   f"**{(test_df['open']==0).sum()}**"])
    lines += _aligned_table(b_headers, b_rows, b_aligns)

    lines += [
        "",
        "---",
        "",
        "## Results",
        "",
        "> Evaluation is always on the **same frozen test set** (20% hold-out).",
        ">",
        "> \"After batch k\" = model has been warm-started through batches 1 … k.",
        "> The single run trains a fresh model on all 2,740 rows at once.",
        "",
    ]

    r_headers = ["Model", "Run", "Batch rows", "Total seen",
                 "Accuracy", "Closed Prec", "Closed Rec", "F1 (closed)", "PR-AUC (closed)"]
    r_aligns  = ["l", "l", "r", "r", "r", "r", "r", "r", "r"]
    r_rows: list[list[str]] = []

    for model_tag, inc_res, single_res in all_results:
        display_name = model_names.get(model_tag, model_tag)
        for bi, r in enumerate(inc_res, 1):
            r_rows.append([
                display_name, f"After batch {bi}",
                str(r["n_train"]), str(r["total_rows_seen"]),
                f"{r['accuracy']:.4f}", f"{r['closed_precision']:.4f}",
                f"{r['closed_recall']:.4f}", f"{r['f1_closed']:.4f}",
                f"{r['pr_auc']:.4f}",
            ])
        r_rows.append([
            f"**{display_name}**", "**Single run (all at once)**",
            str(single_res["n_train"]), str(single_res["total_rows_seen"]),
            f"**{single_res['accuracy']:.4f}**",
            f"{single_res['closed_precision']:.4f}",
            f"{single_res['closed_recall']:.4f}",
            f"{single_res['f1_closed']:.4f}",
            f"{single_res['pr_auc']:.4f}",
        ])
        r_rows.append([""] * 9)  # blank spacer row

    lines += _aligned_table(r_headers, r_rows, r_aligns)

    # ── Key Comparison: batch 5 (all data seen) vs single run ─────────────────
    lines += [
        "",
        "---",
        "",
        "## Key Comparison — After All Batches vs Single Run",
        "",
        "> Both have seen the same 2,740 training rows.",
        "> Incremental = warm-started through 5 batches.  Single = trained on all at once.",
        "",
    ]

    k_headers = ["Model", "Metric", "After batch 5", "Single run", "Δ (single − inc)"]
    k_aligns  = ["l", "l", "r", "r", "r"]
    k_rows: list[list[str]] = []

    for model_tag, inc_res, single_res in all_results:
        display_name = model_names.get(model_tag, model_tag)
        last_inc = inc_res[-1]  # after batch 5
        for metric_key, metric_label in [
            ("accuracy",         "Accuracy"),
            ("closed_precision", "Closed Prec"),
            ("closed_recall",    "Closed Rec"),
            ("f1_closed",        "F1 (closed)"),
            ("pr_auc",           "PR-AUC (closed)"),
        ]:
            inc_val    = last_inc[metric_key]
            single_val = single_res[metric_key]
            delta      = single_val - inc_val
            k_rows.append([
                display_name if metric_label == "Accuracy" else "",
                metric_label,
                f"{inc_val:.4f}",
                f"{single_val:.4f}",
                f"{delta:+.4f}",
            ])

    lines += _aligned_table(k_headers, k_rows, k_aligns)

    # ── Notes ─────────────────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Notes on Results",
        "",
        "### What the experiment shows",
        "",
        "- **Incremental (warm-start)**: the model accumulates knowledge across",
        "  batches.  After batch 5 it has seen all 2,740 training rows, but via",
        "  5 sequential warm-start steps rather than one big `.fit()` call.",
        "- **Single run**: a fresh model trained on all 2,740 rows at once.",
        "- The Δ column shows how the single run compares to the fully",
        "  warm-started model (after all 5 batches).",
        "",
        "### Interpretation",
        "",
        "- **Positive Δ** = single run outperforms warm-start → training on all",
        "  data simultaneously gives the optimizer a better global view.",
        "- **Negative Δ** = warm-start outperforms single run → sequential",
        "  exposure helped (possible regularisation effect, or the optimizer",
        "  found a better local minimum via the warm-start trajectory).",
        "- **Near-zero Δ** = both strategies are equivalent.",
        "",
        "### How warm-start differs from training on each batch alone",
        "",
        "- **Batch-alone** (previous experiment): each batch trains a fresh model.",
        "  The model only ever sees ~548 rows.  Performance is unstable because",
        "  each batch has only ~50 closed examples.",
        "- **Warm-start** (this experiment): the model carries forward its learned",
        "  parameters.  By batch 5 it has been trained on all 2,740 rows",
        "  sequentially.  Performance should converge toward the single-run result.",
        "",
        "---",
        "",
        "## File Artifacts",
        "",
        "| File | Description |",
        "|:-----|:------------|",
        "| `data/project_c_samples/batches/test_set.csv` | Fixed 20% held-out test set |",
        "| `data/project_c_samples/batches/batch_1.csv` … `batch_5.csv` | Individual stratified training batches |",
        "| `data/project_c_samples/batches/all_batches_combined.csv` | All 5 batches combined (single-run input) |",
        "",
    ]

    md_path = RESULTS_DIR / "incremental_results.md"
    md_path.write_text("\n".join(lines))
    print(f"\n{'='*70}")
    print(f"Results written → {md_path}")
    print("Done.")


if __name__ == "__main__":
    main()
