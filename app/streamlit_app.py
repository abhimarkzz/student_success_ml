from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xai import shap_table

st.set_page_config(page_title="Student Success ML", page_icon="🎓", layout="wide")
st.title("🎓 AI-Driven Student Performance & Placement Prediction")
st.caption("ML core: CatBoost + SHAP + model-based what-if analysis")

perf_path = ROOT / "models/performance_mat_regressor.cbm"
pass_path = ROOT / "models/performance_mat_pass_classifier.cbm"
placement_models = list((ROOT / "models").glob("placement_*.cbm"))

if not perf_path.exists():
    st.warning("Performance model not found. Train it first with: python scripts/train_performance.py --subject mat --tune-trials 20")
    st.stop()

model = CatBoostRegressor()
model.load_model(perf_path)
features = model.get_feature_names()

st.subheader("Student profile")
cols = st.columns(3)
example_defaults = {
    "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
    "Pstatus": "T", "Medu": 4, "Fedu": 4, "Mjob": "other", "Fjob": "other",
    "reason": "course", "guardian": "mother", "traveltime": 2, "studytime": 2,
    "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no", "activities": "no",
    "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
    "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 1, "health": 5,
    "absences": 4,
}

row = {}
for i, f in enumerate(features):
    with cols[i % 3]:
        default = example_defaults.get(f, 0)
        if isinstance(default, str):
            row[f] = st.text_input(f, value=default)
        else:
            row[f] = st.number_input(f, value=float(default), step=1.0)

X = pd.DataFrame([row], columns=features)
for c in X.columns:
    if X[c].dtype == object:
        X[c] = X[c].astype(str)

pred = float(model.predict(X)[0])
st.metric("Predicted final grade (G3)", f"{pred:.2f} / 20")

if pass_path.exists():
    clf = CatBoostClassifier()
    clf.load_model(pass_path)
    pass_prob = float(clf.predict_proba(X)[0, 1])
    st.metric("Model-estimated pass probability", f"{pass_prob*100:.1f}%")

st.subheader("Why did the model predict this?")
try:
    table = shap_table(model, X, 12)
    st.dataframe(table, use_container_width=True, hide_index=True)
except Exception as exc:
    st.info(f"SHAP explanation unavailable: {exc}")

st.subheader("What-if simulation")
left, right = st.columns(2)
with left:
    studytime = st.number_input("Scenario studytime (1–4)", min_value=1.0, max_value=4.0, value=float(row.get("studytime", 2)))
with right:
    absences = st.number_input("Scenario absences", min_value=0.0, max_value=93.0, value=float(row.get("absences", 4)))
scenario = X.copy()
if "studytime" in scenario:
    scenario.loc[0, "studytime"] = studytime
if "absences" in scenario:
    scenario.loc[0, "absences"] = absences
scenario_pred = float(model.predict(scenario)[0])
st.metric("Scenario predicted grade", f"{scenario_pred:.2f} / 20", delta=f"{scenario_pred-pred:+.2f}")
st.caption("The what-if result is a model scenario, not a causal guarantee.")

st.subheader("Placement model")
if placement_models:
    st.success(f"Placement model available: {placement_models[0].name}")
    st.caption("Use the corresponding placement dataset schema; do not mix features from the performance model unless the training data supports them.")
else:
    st.info("No placement model is trained yet. Use the placement training command in the README with a public benchmark or, preferably, permissioned anonymized institutional data.")
