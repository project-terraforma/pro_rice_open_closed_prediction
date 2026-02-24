# Data Dictionary: Open vs Closed Place Prediction

This document explains each column in the parquet dataset and how it might help predict `open` vs `closed`. It also captures clarifications discussed during exploration.

## Big Picture
- The parquet file is the **source of truth** for modeling. The online schema is useful context, but the model must follow what is **actually present in the parquet**.
- `open` is the **label** (target). All other columns are **features** or metadata.
- `confidence` is **not the label**. It is a provider-side confidence signal and can be constant for some providers.
- `sources` provides provenance. If a property is not explicitly listed in `sources.property`, its provenance falls back to the **root source** (the entry with `property: ""`) for that record.

## Columns

### `id`
**What it is:** Unique identifier for the place record.
**Why it matters:** Identifier only.
**Use for modeling:** **No** (never use as a feature).

### `geometry`
**What it is:** Encoded point location (WKB binary). Represents latitude/longitude.
**Why it matters:** Location can correlate with open/closed if you derive spatial features.
**Use for modeling:** **Maybe later** (requires decoding to lat/lon).

### `bbox`
**What it is:** Bounding box around the location (min/max lat/lon).
**Why it matters:** Usually redundant with geometry; could reflect geocoding precision.
**Use for modeling:** **Unlikely**, unless you derive spatial/precision features.

### `type`
**What it is:** Record type (typically "place").
**Why it matters:** Likely constant.
**Use for modeling:** **No**.

### `version`
**What it is:** Schema/record version.
**Why it matters:** Could reflect pipeline changes or data freshness if multiple versions appear.
**Use for modeling:** **Maybe**, if there is variation.

### `sources`
**What it is:** List of provenance objects for data sources. Each entry typically has:
- `dataset`: data provider (e.g., Meta, Microsoft)
- `confidence`: per-source confidence
- `property`: JSON Pointer for which property that source asserts (e.g., `/properties/existence`)
- `record_id`: source-specific ID
- `update_time`: source update time

**Why it matters:** More sources and more recent updates can indicate a better-supported, current record.
**Use for modeling:** **Yes** for source count/recency/diversity. Use per-source confidence only as an experimental feature unless semantics are fully confirmed.

**Important provenance rule:**
- If a property has an explicit `sources.property`, it is attributed to that source.
- If a property does **not** have an explicit pointer, its provenance falls back to the **root source** (the entry with `property: ""`) for that record.
- Multiple sources **can** point to the same property, meaning multiple datasets contributed evidence.

### `names`
**What it is:** Place name info (primary, common, rules).
**Why it matters:** Named, branded places may be more likely to be open; missing/odd names can be a weak negative signal.
**Use for modeling:** **Maybe** (length, missingness, brand-like patterns).

### `categories`
**What it is:** Primary category plus alternate categories.
**Why it matters:** Some categories may have higher closure rates or seasonal behavior.
**Use for modeling:** **Yes** (category features are often strong).

### `confidence`
**What it is:** Provider-side confidence related to finding the place at the location and whether it is in business/open.
**Why it matters:** It can be predictive, but it may encode provider-specific behavior and is constant for some providers.
**Use for modeling:** **Not recommended for the core baseline** (per Overture guidance). Keep only as optional/ablation.

**Clarification:** This is **not** "open right now". It is about existence/validity of the place record.

### `websites`
**What it is:** List of website URLs.
**Why it matters:** Active businesses often have websites.
**Use for modeling:** **Yes** (presence/absence, count).

### `socials`
**What it is:** List of social media URLs.
**Why it matters:** Active social presence can indicate an open/active place.
**Use for modeling:** **Yes** (presence/absence).

### `phones`
**What it is:** List of phone numbers.
**Why it matters:** Active businesses often provide phones.
**Use for modeling:** **Yes** (presence/absence, count).

### `brand`
**What it is:** Brand metadata (if place belongs to a chain).
**Why it matters:** Branded chain locations are more likely to be maintained and open; missing brand doesn’t imply closed.
**Use for modeling:** **Yes** (presence/absence).

### `addresses`
**What it is:** Structured address objects (country, region, locality, etc.).
**Why it matters:** Missing addresses can be a red flag; region/locality features might matter later.
**Use for modeling:** **Maybe** (presence or location-based features).

### `open`
**What it is:** Label (1 = open, 0 = closed).
**Why it matters:** This is the target you’re predicting.
**Use for modeling:** **No** (only for training/evaluation).

## Open/Closed Meaning
- The dataset uses `open` as a **business operating status** label (open vs closed), **not time-of-day** hours.
- This can include **temporarily closed** or **permanently closed** in the single “closed” label if upstream data collapsed them.

## Data Lineage / Snapshot Assumptions
- The parquet file is likely a **snapshot** of a larger, continuously updated system.
- `sources.update_time` reflects when sources last updated, not necessarily the snapshot time.
- Open/closed labels were produced in **December 2025** by an external provider using agents/web search and may contain some noise.
- There is no explicit snapshot timestamp column in the delivered sample.

## Practical Modeling Takeaways
- Keep `confidence` out of the primary baseline unless explicitly approved for use.
- Use `sources` to extract: source count, recency, and dataset diversity.
- Use contact/brand presence as lighter‑weight signals.
- Avoid using identifiers (`id`) directly.

## Confirmed with Overture Team (Feb 2026)
- Open/closed labels come from an external provider that uses agents/web search.
- Label noise is possible (not perfectly clean ground truth).
- Class imbalance around ~90/10 open/closed is close to production and should be treated as part of the real task.
- No larger labeled open/closed set is currently available.
- Upstream provider signals are not currently accessible for this project.
- From Overture's side, there are currently no constraints on trying external data sources.

## Still Worth Clarifying
- Exact semantics/calculation of `sources[*].confidence` across providers.
- Any known edge cases in `sources.property` merging behavior.

## Important Dataset Scope Note
- `data/matching_validation/samples_3k_project_c_updated.csv` is **not** an open/closed dataset.
- It is a place-matching validation dataset:
  - `label = 1` means the pair matches.
  - `label = 0` means the pair does not match.
  - `base_*` fields correspond to one side of the pair.

## Exploration Findings (Training Split Only)
These findings are from `data/train_split.parquet` and may not represent val/test splits.

### Latest `sources.update_time`
**Finding:** The latest `sources.update_time` observed in training data is `2025-02-24T08:00:00.000Z`.  
**Why it matters:** Suggests the snapshot is **no earlier than** this timestamp, but does **not** prove exact snapshot time.

**Command used:**
```bash
python -c "import pandas as pd, pyarrow.parquet as pq; df=pq.read_table('data/train_split.parquet').to_pandas();\n\ndef max_update(sources):\n    if sources is None: return None\n    try:\n        times=[s.get('update_time') for s in sources if s.get('update_time')]\n        return max(times) if times else None\n    except Exception:\n        return None\n\nupdates=df['sources'].apply(max_update)\nupdates=updates.dropna()\nprint('max_update_time', updates.max())\n"
```

### Version Values in Training Data
**Finding:** Only `version = 0` appears in the training split (all 2,397 rows).  
**Why it matters:** `version` will not be a useful feature in training data as it has no variation.

**Command used:**
```bash
python -c "import pyarrow.parquet as pq; import pandas as pd; df=pq.read_table('data/train_split.parquet', columns=['version']).to_pandas(); print(df['version'].value_counts().sort_index())"
```

### Batch-Like Update Pattern (Inference)
**Finding:** The same `sources.update_time` repeats across many rows in training data, suggesting batch updates.  
**Why it matters:** Indicates updates may arrive in batches rather than continuous per-record refreshes.

**How we inferred it:** Observed repeated update timestamps when scanning the sources in training data. This is an inference, not a confirmed fact.

### Recency Distribution (Training Split)
**Finding:** Using the latest `sources.update_time` as a snapshot proxy, most records are very recent. Recency percentiles (days since snapshot proxy) are:  
0%: 0, 10%: 0, 25%: 0, 50%: 0, 75%: 0, 90%: 154, 95%: 1527, 99%: 3763, 100%: 5821.  
**Why it matters:** A large portion of records have update times equal to the snapshot proxy, but a long tail is many years old. This can inform “stale” thresholds.

**Command used:**
```bash
python -c "import pandas as pd, pyarrow.parquet as pq; df=pq.read_table('data/train_split.parquet').to_pandas();\n\ndef max_update(sources):\n    if sources is None: return None\n    try:\n        times=[s.get('update_time') for s in sources if s.get('update_time')]\n        return max(times) if times else None\n    except Exception:\n        return None\n\nmax_updates=df['sources'].apply(max_update).dropna()\nmax_updates=pd.to_datetime(max_updates, errors='coerce').dropna()\nif len(max_updates)==0:\n    print('no update_time values')\n    raise SystemExit(0)\n\nsnapshot=max_updates.max()\nrecency_days=(snapshot - max_updates).dt.days\nprint('snapshot_proxy', snapshot.isoformat())\nprint('recency_days_percentiles')\nfor p in [0, 10, 25, 50, 75, 90, 95, 99, 100]:\n    val=recency_days.quantile(p/100)\n    print(f'{p:>3}% {val:.0f}')\n"
```
