"""Persist Alex filtered datasets in model-compatible parquet form."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_alex_transfer_eval import _load_alex_eval_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Alex filtered city datasets as persisted parquet artifacts."
    )
    parser.add_argument(
        "--alex-assets-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "places-status-engine" / "assets",
        help="Directory containing Alex raw and labeled city parquet files.",
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
    city_frames = []

    for city in args.cities:
        city_df = _load_alex_eval_data(args.alex_assets_dir, [city]).reset_index(drop=True)
        city_path = args.output_dir / f"alex_{city}_filtered.parquet"
        city_df.to_parquet(city_path, index=False)
        city_frames.append(city_df)
        manifest_rows.append(
            {
                "city": city,
                "rows": int(len(city_df)),
                "open_rows": int(city_df["open"].sum()),
                "closed_rows": int((1 - city_df["open"]).sum()),
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
