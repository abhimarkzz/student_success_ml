# START HERE — 10-minute run

## A. Real UCI performance model

```bash
cd student_success_ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_datasets.py --uci
python scripts/train_performance.py --subject mat --tune-trials 20
```

Then explain a student:

```bash
python scripts/explain_student.py \
  --model models/performance_mat_regressor.cbm \
  --input examples/student_mat_example.csv
```

Run what-if:

```bash
python scripts/what_if.py \
  --model models/performance_mat_regressor.cbm \
  --input examples/student_mat_example.csv \
  --changes '{"studytime": 4, "absences": 2}'
```

Start UI:

```bash
streamlit run app/streamlit_app.py
```

## B. Placement model

For an immediate reproducible public benchmark:

```bash
python scripts/download_datasets.py --placement-benchmark
python scripts/train_placement.py --csv data/raw/placement/train.csv --target PlacementStatus
```

For your final academic project, replace the benchmark with **permissioned anonymized institutional placement data** while preserving the target and feature schema.

## C. What to put into your presentation

Only use the metrics actually produced by your real training run. Do not use the smoke-test numbers or any numbers from this README.

Suggested methodological statement:

> “We use CatBoost gradient-boosted decision trees as the primary tabular learner because the student profile contains mixed numerical and categorical attributes. The performance pipeline excludes G1/G2 from the early-warning feature set to avoid the temporal leakage issue documented by UCI. SHAP provides local explanations, while the what-if module reports model-based counterfactual scenarios.”
