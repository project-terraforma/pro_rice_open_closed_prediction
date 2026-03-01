"""Validate feature inventory and bundle tier constraints.

Usage:
    python src/models/validate_feature_bundles.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = PROJECT_ROOT / "docs" / "feature_inventory.csv"
BUNDLES_PATH = PROJECT_ROOT / "docs" / "feature_bundles.json"


def load_inventory(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required_cols = {
        "feature_name",
        "source_field",
        "transform",
        "cost_tier",
        "allowed_under_KR1.2",
    }
    missing = required_cols - set(rows[0].keys()) if rows else required_cols
    if missing:
        raise ValueError(f"Inventory missing columns: {sorted(missing)}")

    inventory: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row["feature_name"].strip()
        if name in inventory:
            raise ValueError(f"Duplicate inventory feature: {name}")
        inventory[name] = row
    return inventory


def load_bundles(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    bundles = data.get("bundles")
    if not isinstance(bundles, dict):
        raise ValueError("feature_bundles.json must contain top-level 'bundles' object")
    return {k: list(v) for k, v in bundles.items()}


def validate(inventory: dict[str, dict[str, str]], bundles: dict[str, list[str]]) -> list[str]:
    errors: list[str] = []

    required_bundle_names = {"low_only", "low_plus_medium", "full_schema_native"}
    missing_bundle_names = required_bundle_names - set(bundles)
    if missing_bundle_names:
        errors.append(f"Missing required bundles: {sorted(missing_bundle_names)}")

    for bundle_name, features in bundles.items():
        seen = set()
        for feature in features:
            if feature in seen:
                errors.append(f"Duplicate feature '{feature}' in bundle '{bundle_name}'")
            seen.add(feature)
            if feature not in inventory:
                errors.append(f"Feature '{feature}' in bundle '{bundle_name}' not found in inventory")
                continue

            row = inventory[feature]
            if row["allowed_under_KR1.2"].strip().lower() != "yes":
                errors.append(f"Feature '{feature}' in bundle '{bundle_name}' is not KR1.2-allowed")

            tier = row["cost_tier"].strip().lower()
            if bundle_name == "low_only" and tier != "low":
                errors.append(f"Feature '{feature}' in low_only has non-low tier '{tier}'")
            if bundle_name == "low_plus_medium" and tier not in {"low", "medium"}:
                errors.append(f"Feature '{feature}' in low_plus_medium has invalid tier '{tier}'")

    if all(name in bundles for name in ["low_only", "low_plus_medium", "full_schema_native"]):
        low_only = set(bundles["low_only"])
        low_plus_medium = set(bundles["low_plus_medium"])
        full_schema_native = set(bundles["full_schema_native"])

        if not low_only.issubset(low_plus_medium):
            missing = sorted(low_only - low_plus_medium)
            errors.append(f"low_only must be subset of low_plus_medium; missing: {missing}")

        if not low_plus_medium.issubset(full_schema_native):
            missing = sorted(low_plus_medium - full_schema_native)
            errors.append(
                "low_plus_medium must be subset of full_schema_native; "
                f"missing: {missing}"
            )

    return errors


def main() -> None:
    inventory = load_inventory(INVENTORY_PATH)
    bundles = load_bundles(BUNDLES_PATH)

    errors = validate(inventory, bundles)
    if errors:
        print("Feature bundle validation FAILED:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("Feature bundle validation PASSED")
    print(f"Inventory features: {len(inventory)}")
    for name, features in bundles.items():
        print(f"- {name}: {len(features)} features")


if __name__ == "__main__":
    main()
