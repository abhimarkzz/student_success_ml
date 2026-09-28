from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--changes", required=True, help='Dict, e.g. \'{"studytime": 4, "absences": 2}\'')
    args = parser.parse_args()

    model = CatBoostRegressor()
    model.load_model(args.model)
    X = pd.read_csv(args.input)

    expected = list(model.feature_names_ or [])
    if not expected:
        expected = list(X.columns)
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise ValueError(f"Input is missing model features: {missing}")
    X = X[expected].copy()

    cat_indices = list(model.get_cat_feature_indices())
    for idx in cat_indices:
        col = expected[idx]
        X[col] = X[col].fillna("Unknown").astype(str)

    base = float(model.predict(X)[0])
    changes = ast.literal_eval(args.changes)
    scenario = X.copy()
    for col, value in changes.items():
        if col not in scenario.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(scenario.columns)}")
        scenario.loc[scenario.index[0], col] = value
        if col in {expected[i] for i in cat_indices}:
            scenario[col] = scenario[col].astype(str)

    counter = float(model.predict(scenario)[0])
    print(f"Baseline model prediction: {base:.3f}")
    print(f"Scenario model prediction: {counter:.3f}")
    print(f"Model delta: {counter - base:+.3f}")
    print("Caution: this is a model-based counterfactual scenario, not a causal effect estimate.")


if __name__ == "__main__":
    main()
