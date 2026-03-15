"""Persist Alex filtered datasets in model-compatible parquet form.

Expected repo-local inputs per city:
- <input-dir>/<city>_places_raw.parquet
- <input-dir>/<city>_places_labeled_checkpoint.parquet
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import pandas as pd


def _point_wkb(lon: float, lat: float) -> bytes:
    return struct.pack("<BI2d", 1, 1, float(lon), float(lat))


def _prepare_alex_eval_df(raw_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    merged = raw_df.merge(labels_df, left_on="id", right_on="overture_id", how="inner")
    merged = merged[merged["fsq_label"].isin(["open", "closed"])].copy()
    merged["open"] = (merged["fsq_label"] == "open").astype(int)
    merged["geometry"] = [
        _point_wkb(lon, lat) for lon, lat in zip(merged["lon"].astype(float), merged["lat"].astype(float))
    ]
    merged["bbox"] = None
    merged["type"] = "place"
    merged["version"] = 0
    return merged


def _load_city_df(input_dir: Path, city: str) -> pd.DataFrame:
    raw_path = input_dir / f"{city}_places_raw.parquet"
    labels_path = input_dir / f"{city}_places_labeled_checkpoint.parquet"
    if not raw_path.exists():
        raise SystemExit(
            f"Missing repo-local raw parquet: {raw_path}\n"
            f"Copy {city}_places_raw.parquet into {input_dir} or pass --input-dir."
        )
    if not labels_path.exists():
        raise SystemExit(
            f"Missing repo-local label parquet: {labels_path}\n"
            f"Copy {city}_places_labeled_checkpoint.parquet into {input_dir} or pass --input-dir."
        )
    city_df = _prepare_alex_eval_df(pd.read_parquet(raw_path), pd.read_parquet(labels_path))
    city_df["alex_city"] = city
    return city_df.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Alex filtered city datasets as persisted parquet artifacts."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "alex_assets",
        help="Repo-local directory containing Alex raw and labeled city parquet files.",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["sf", "nyc"],
        help="Alex datasets to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "alex_filtered_datasets",
        help="Directory where filtered parquet files will be written.",
    )
    parser.add_argument(
        "--write-combined",
        action="store_true",
        help="Also write one combined parquet across all requested cities.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    city_frames: list[pd.DataFrame] = []

    for city in args.cities:
        city_df = _load_city_df(args.input_dir, city)
        city_path = args.output_dir / f"alex_{city}_filtered.parquet"
        city_df.to_parquet(city_path, index=False)
        city_frames.append(city_df)
        manifest_rows.append(
            {
                "city": city,
                "rows": int(len(city_df)),
                "open_rows": int(city_df["open"].sum()),
                "closed_rows": int((1 - city_df["open"]).sum()),
                "input_dir": str(args.input_dir),
                "output_path": str(city_path),
            }
        )
        print(
            f"Wrote {city_path} rows={len(city_df):,} "
            f"open={int(city_df['open'].sum()):,} closed={int((1 - city_df['open']).sum()):,}"
        )

    if args.write_combined and city_frames:
        combined_df = city_frames[0] if len(city_frames) == 1 else pd.concat(city_frames, ignore_index=True)
        combined_path = args.output_dir / "alex_combined_filtered.parquet"
        combined_df.to_parquet(combined_path, index=False)
        print(f"Wrote {combined_path} rows={len(combined_df):,}")

    manifest_path = args.output_dir / "alex_filtered_manifest.json"
    manifest_path.write_text(json.dumps({"datasets": manifest_rows}, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
