from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import REPORT_DIR
from src.xai import shap_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    model = CatBoostRegressor()
    model.load_model(args.model)
    X = pd.read_csv(args.input)
    cat_cols = list(model.get_cat_feature_indices())
    # CatBoost indexes are used for the loaded model; map to names.
    cat_names = [X.columns[i] for i in cat_cols]
    for c in cat_names:
        X[c] = X[c].fillna("Unknown").astype(str)

    pred = float(model.predict(X)[0])
    table = shap_table(model, X, args.top)
    out = REPORT_DIR / "student_local_explanation.csv"
    table.to_csv(out, index=False)
    print(f"Predicted final grade: {pred:.3f}")
    print(table.to_string(index=False))
    print(f"Saved explanation to {out}")


if __name__ == "__main__":
    main()
