"""Basic data discovery for the Open vs Closed dataset.

This script inspects the training split to understand column types,
null patterns, nested structures, and simple signal strength stats.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DATA_PATH = Path("data/train_split.parquet")


def has_value(x) -> int:
    if x is None:
        return 0
    if isinstance(x, float) and np.isnan(x):
        return 0
    try:
        return 1 if len(x) > 0 else 0
    except Exception:
        return 1


def count_sources(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def extract_datasets(sources) -> list[str]:
    if sources is None:
        return []
    try:
        return [s.get("dataset") for s in sources]
    except Exception:
        return []


def extract_props(sources) -> list[str]:
    if sources is None:
        return []
    try:
        return [s.get("property") for s in sources]
    except Exception:
        return []


def max_source_conf(sources) -> float:
    if sources is None:
        return float("nan")
    try:
        vals = [s.get("confidence") for s in sources if s.get("confidence") is not None]
        return max(vals) if vals else float("nan")
    except Exception:
        return float("nan")

def format_sources(sources) -> list[dict]:
    if sources is None:
        return []
    formatted = []
    try:
        for s in sources:
            formatted.append(
                {
                    "dataset": s.get("dataset"),
                    "property": s.get("property"),
                    "confidence": s.get("confidence"),
                    "record_id": s.get("record_id"),
                    "update_time": s.get("update_time"),
                }
            )
    except Exception:
        return []
    return formatted


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing parquet file: {DATA_PATH}")

    df = pq.read_table(DATA_PATH).to_pandas()

    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Columns ===")
    print(df.columns.tolist())

    print("\n=== Dtypes ===")
    print(df.dtypes)

    print("\n=== Label distribution ===")
    print(df["open"].value_counts())

    print("\n=== Example rows (first 2) ===")
    print(df.head(7).to_dict(orient="records"))

    # print("\n=== Sources only (next 5 rows) ===")
    # for idx, row in df.iloc[2:7].iterrows():
    #     print(f"row_index={idx}")
    #     print(format_sources(row["sources"]))
    #     print("---")

    # Presence flags
    for col in ["websites", "phones", "socials", "brand", "sources", "addresses", "names", "categories"]:
        df[f"{col}_present"] = df[col].apply(has_value)

    df["sources_n"] = df["sources"].apply(count_sources)
    df["datasets"] = df["sources"].apply(extract_datasets)
    df["props"] = df["sources"].apply(extract_props)
    df["max_source_conf"] = df["sources"].apply(max_source_conf)

    print("\n=== Confidence summary ===")
    print(df["confidence"].describe())

    print("\n=== Presence rates by label ===")
    for col in ["websites_present", "phones_present", "socials_present", "brand_present"]:
        rates = df.groupby("open")[col].mean()
        print(col, rates.to_dict())

    print("\n=== Sources count by label ===")
    print(df.groupby("open")["sources_n"].describe())

    print("\n=== Max source confidence by label ===")
    print(df.groupby("open")["max_source_conf"].mean())

    print("\n=== Dataset names by label (top 5) ===")
    for label in [0, 1]:
        c = Counter()
        for lst in df[df["open"] == label]["datasets"]:
            for d in lst:
                c[d] += 1
        print("label", label, c.most_common(5))

    print("\n=== Properties by label (top 5) ===")
    for label in [0, 1]:
        c = Counter()
        for lst in df[df["open"] == label]["props"]:
            for d in lst:
                c[d] += 1
        print("label", label, c.most_common(5))

    print("\n=== Simple closed-rate slices ===")
    def closed_rate(mask):
        sub = df[mask]
        if len(sub) == 0:
            return 0.0, 0
        return 1.0 - sub["open"].mean(), len(sub)

    cases = {
        "no_web": df["websites_present"] == 0,
        "no_phone": df["phones_present"] == 0,
        "no_web_and_phone": (df["websites_present"] == 0) & (df["phones_present"] == 0),
        "low_conf_lt_0_7": df["confidence"] < 0.7,
        "low_conf_lt_0_7_no_web_phone": (df["confidence"] < 0.7)
        & (df["websites_present"] == 0)
        & (df["phones_present"] == 0),
        "single_source": df["sources_n"] == 1,
        "multi_source": df["sources_n"] >= 2,
    }

    for name, mask in cases.items():
        closed, n = closed_rate(mask)
        print(name, "n", n, "closed_rate", round(closed, 3))


if __name__ == "__main__":
    main()

# === Shape ===
# (2397, 15)

# === Columns ===
# ['id', 'geometry', 'bbox', 'type', 'version', 'sources', 'names', 'categories', 'confidence', 'websites', 'socials', 'phones', 'brand', 'addresses', 'open']

# === Dtypes ===
# id                str
# geometry       object
# bbox           object
# type              str
# version         int32
# sources        object
# names          object
# categories     object
# confidence    float64
# websites       object
# socials        object
# phones         object
# brand          object
# addresses      object
# open            int64
# dtype: object

# === Label distribution ===
# open
# 1    2178
# 0     219
# Name: count, dtype: int64

# === Example rows (first 2) ===
# [{'id': '08f44a8cf0ace46e03a59ad9ba77dd07', 'geometry': b'\x01\x01\x00\x00\x00h\x985\x0c\xfa\x14T\xc0I\x14\xb5\x8fpm;@', 'bbox': {'xmax': -80.3277587890625, 'xmin': -80.32777404785156, 'ymax': 27.427501678466797, 'ymin': 27.42749786376953}, 'type': 'place', 'version': 0, 'sources': array([{'confidence': 0.9793990828827596, 'dataset': 'meta', 'property': '', 'record_id': '409645426098970', 'update_time': '2025-02-24T08:00:00.000Z'},
#        {'confidence': 0.77, 'dataset': 'Microsoft', 'property': '/properties/existence', 'record_id': '562949966440142', 'update_time': '2024-12-27T19:31:13.220Z'}],
#       dtype=object), 'names': {'common': None, 'primary': 'Family Dollar', 'rules': None}, 'categories': {'alternate': array(['arts_and_crafts'], dtype=object), 'primary': 'discount_store'}, 'confidence': 0.9952617890630348, 'websites': array(['https://locations.familydollar.com/fl/fort-pierce/2047-s-us-highway-1'],
#       dtype=object), 'socials': array(['https://www.facebook.com/409645426098970'], dtype=object), 'phones': array(['+17722520436'], dtype=object), 'brand': {'names': {'common': None, 'primary': 'Family Dollar', 'rules': None}, 'wikidata': None}, 'addresses': array([{'country': 'US', 'freeform': '2047 S US Highway 1', 'locality': 'Fort Pierce', 'postcode': '34950-5149', 'region': 'FL'}],
#       dtype=object), 'open': 0}, {'id': '08f2a100ee00e04a034133e5a32880e4', 'geometry': b'\x01\x01\x00\x00\x00\x8b\x07J\xc0\x9etR\xc0\x17\x87\x8ex\r]D@', 'bbox': {'xmax': -73.82218933105469, 'xmin': -73.82219696044922, 'ymax': 40.72697448730469, 'ymin': 40.726966857910156}, 'type': 'place', 'version': 0, 'sources': array([{'confidence': 0.9793990828827596, 'dataset': 'meta', 'property': '', 'record_id': '393047740772122', 'update_time': '2025-02-24T08:00:00.000Z'}],
#       dtype=object), 'names': {'common': None, 'primary': 'Sandwich Bar', 'rules': None}, 'categories': {'alternate': array(['fast_food_restaurant', 'middle_eastern_restaurant'], dtype=object), 'primary': 'sandwich_shop'}, 'confidence': 0.9793990828827596, 'websites': array(['https://www.toasttab.com/sandwich-bar-71-32-main-st/v3/?mode=fulfillment'],
#       dtype=object), 'socials': array(['https://www.facebook.com/393047740772122'], dtype=object), 'phones': array(['+17185441014'], dtype=object), 'brand': None, 'addresses': array([{'country': 'US', 'freeform': '71-32 Main St', 'locality': 'New York', 'postcode': '11367', 'region': 'NY'}],
#       dtype=object), 'open': 1}]

# === Confidence summary ===
# count    2397.000000
# mean        0.859394
# std         0.184791
# min         0.196172
# 25%         0.770000
# 50%         0.935964
# 75%         0.979399
# max         0.995388
# Name: confidence, dtype: float64

# === Presence rates by label ===
# websites_present {0: 0.8127853881278538, 1: 0.8953168044077136}
# phones_present {0: 0.8767123287671232, 1: 0.9715335169880625}
# socials_present {0: 0.7945205479452054, 1: 0.8296602387511478}
# brand_present {0: 0.1324200913242009, 1: 0.17263544536271808}

# === Sources count by label ===
#        count      mean       std  min  25%  50%  75%  max
# open                                                     
# 0      219.0  1.123288  0.329520  1.0  1.0  1.0  1.0  2.0
# 1     2178.0  1.227732  0.419465  1.0  1.0  1.0  1.0  2.0

# === Max source confidence by label ===
# open
# 0    0.750162
# 1    0.863589
# Name: max_source_conf, dtype: float64

# === Dataset names by label (top 5) ===
# label 0 [('meta', 175), ('Microsoft', 71)]
# label 1 [('meta', 1822), ('Microsoft', 852)]

# === Properties by label (top 5) ===
# label 0 [('', 219), ('/properties/existence', 27)]
# label 1 [('', 2178), ('/properties/existence', 496)]

# === Simple closed-rate slices ===
# no_web n 269 closed_rate 0.152
# no_phone n 89 closed_rate 0.303
# no_web_and_phone n 49 closed_rate 0.367
# low_conf_lt_0_7 n 276 closed_rate 0.221
# low_conf_lt_0_7_no_web_phone n 36 closed_rate 0.389
# single_source n 1874 closed_rate 0.102
# multi_source n 523 closed_rate 0.052
