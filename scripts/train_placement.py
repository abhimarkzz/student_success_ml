from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import MODEL_DIR, REPORT_DIR, save_json
from src.placement import load_placement_csv, train_placement_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="PlacementStatus")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    bundle = load_placement_csv(csv_path, args.target)
    model, metrics = train_placement_model(bundle)

    stem = csv_path.stem
    model_path = MODEL_DIR / f"placement_{stem}.cbm"
    model.save_model(model_path)

    importance = pd.DataFrame({"feature": bundle.X.columns, "importance": model.get_feature_importance()}).sort_values("importance", ascending=False)
    importance.to_csv(REPORT_DIR / f"placement_{stem}_feature_importance.csv", index=False)
    save_json(REPORT_DIR / f"placement_{stem}_metrics.json", {
        "dataset_file": str(csv_path),
        "target": bundle.target,
        "model": "CatBoostClassifier",
        "metrics": metrics,
        "categorical_features": bundle.cat_cols,
        "warning": "Only describe the dataset provenance exactly as documented by its source.",
    })
    print(metrics)
    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
