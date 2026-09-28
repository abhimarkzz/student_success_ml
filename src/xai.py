from __future__ import annotations

from pathlib import Path

import pandas as pd
import shap


def shap_table(model, X: pd.DataFrame, max_rows: int = 20) -> pd.DataFrame:
    """Return one-row local SHAP contributions with feature values."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[0]
    values = values if getattr(values, "ndim", 1) == 2 else values.reshape(1, -1)
    row = values[0]
    out = pd.DataFrame({
        "feature": X.columns,
        "value": X.iloc[0].astype(str).values,
        "shap_value": row,
        "abs_shap": abs(row),
    }).sort_values("abs_shap", ascending=False)
    return out.head(max_rows).reset_index(drop=True)


def save_shap_summary(model, X: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    table = shap_table(model, X, top_n)
    table.to_csv(path, index=False)
