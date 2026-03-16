from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SF_NY_DIR = ROOT / "data" / "sf_ny"

FILES = [
    "alex_combined_filtered.parquet",
    "alex_nyc_filtered.parquet",
    "alex_sf_filtered.parquet",
]


def main():
    print("Converting alex filtered parquet files to CSV in:", SF_NY_DIR)
    for fname in FILES:
        p = SF_NY_DIR / fname
        if not p.exists():
            print(f"  Skipping missing file: {p.name}")
            continue
        try:
            df = pd.read_parquet(p)
            out = p.with_suffix('.csv')
            df.to_csv(out, index=False)
            print(f"  Converted: {p.name} -> {out.name} ({len(df):,} rows)")
        except Exception as e:
            print(f"  Failed to convert {p.name}: {e}")


if __name__ == '__main__':
    main()
