# AI-Driven Student Performance & Placement Prediction System — ML Core

This repository is the **ML implementation layer** for the B.Tech project:

> **AI-Driven Student Performance & Placement Prediction System**
> From Prediction to Explainable and Personalized Student Success

## What is implemented

### 1. Student performance model — verified public UCI data
- **Primary model:** CatBoostRegressor
- Predicts final grade `G3` from **early/interpretable features only**.
- `G1` and `G2` are intentionally excluded from the main model because UCI warns that they are strongly correlated with `G3`; using them for an early-warning system would create an unrealistically easy prediction setting.
- Optional derived pass-risk classifier (pass = G3 >= 10) is also trained.
- SHAP explanations are generated for global feature importance and individual student explanations.
- A controlled what-if module is included for model-based scenario simulation.

### 2. Placement model — public benchmark adapter
- **Primary model:** CatBoostClassifier
- Reads any placement CSV with a documented target column; an included downloader can fetch the public Hugging Face `Krooz/Campus_Recruitment_CSV` benchmark.
- This benchmark is described as placement data for a "XYZ campus" and exposes 10 placement-related fields plus `PlacementStatus`. Its repository license is CC0. **Do not call it institutionally verified real-world student data.**
- For an academically stronger final project, replace it with your college's **anonymized, permissioned placement data** and keep the same training pipeline.

## Why CatBoost here?
CatBoost is a gradient-boosted decision-tree library designed to work directly with categorical features. Its ordered boosting and categorical-feature processing were introduced specifically to reduce prediction shift/target-leakage problems in boosting. This matches the mixed tabular structure of student data and reduces the need for fragile one-hot preprocessing.

It is not claimed to be universally "the best model". In this repository it is the **best-fit primary model for the current project constraints**, and training includes validation-based tuning so measured results can be reported honestly.

## Verified dataset links

### UCI Student Performance — primary real/public benchmark
Official UCI page:
https://archive.ics.uci.edu/dataset/320/student+performance

Verified facts from UCI:
- 649 records
- 30 official features
- classification and regression tasks
- data from two Portuguese secondary schools
- academic, demographic, social and school-related attributes
- `G3` is the final grade; `G1` and `G2` are earlier-period grades
- UCI explicitly warns about the strong correlation between `G1`/`G2` and `G3`

### OULAD — optional next-stage real learning-analytics benchmark
https://analyse.kmi.open.ac.uk/open_dataset

### Public placement benchmark used by the adapter
Hugging Face:
https://huggingface.co/datasets/Krooz/Campus_Recruitment_CSV

The benchmark currently exposes train/validation/test splits and placement attributes such as CGPA, internships, projects, certifications, aptitude, soft skills, extracurricular activity and placement training.

## Folder structure

```text
student_success_ml/
├── app/
│   └── streamlit_app.py
├── configs/
│   └── placement_config.json
├── data/
│   ├── raw/                  # downloaded datasets go here
│   ├── README.md
│   └── processed/
├── models/
├── reports/
├── scripts/
│   ├── download_datasets.py
│   ├── train_performance.py
│   ├── train_placement.py
│   ├── explain_student.py
│   └── what_if.py
├── src/
│   ├── common.py
│   ├── performance.py
│   ├── placement.py
│   └── xai.py
├── requirements.txt
└── README.md
```

## Run on your machine

### macOS / Linux

```bash
cd student_success_ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```powershell
cd student_success_ml
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Download the datasets

For the UCI Student Performance dataset:

```bash
python scripts/download_datasets.py --uci
```

For the public placement benchmark:

```bash
python scripts/download_datasets.py --placement-benchmark
```

Or download only UCI manually from the official page and place `student-mat.csv` and/or `student-por.csv` in `data/raw/uci_student_performance/`.

## Train the real UCI performance model

```bash
python scripts/train_performance.py --subject mat --tune-trials 20
```

or:

```bash
python scripts/train_performance.py --subject por --tune-trials 20
```

The script creates:
- CatBoost regression model (`.cbm`)
- CatBoost pass-risk model (`.cbm`)
- metrics JSON
- feature-importance CSV
- model metadata JSON

**Important:** the generated metrics are real measured results from your run. The repository does not contain invented accuracy/F1/RMSE values.

## Train the placement model

```bash
python scripts/train_placement.py --csv data/raw/placement/train.csv
```

You can use your own institutional CSV instead:

```bash
python scripts/train_placement.py \
  --csv /path/to/your_anonymized_placement_data.csv \
  --target PlacementStatus
```

The training script automatically detects categorical columns and keeps the target out of the feature set.

## Explain a student prediction

After training:

```bash
python scripts/explain_student.py \
  --model models/performance_mat_regressor.cbm \
  --input examples/student_mat_example.csv
```

It reports the predicted final grade and the strongest positive/negative SHAP contributions.

## Run a what-if scenario

```bash
python scripts/what_if.py \
  --model models/performance_mat_regressor.cbm \
  --input examples/student_mat_example.csv \
  --changes '{"studytime": 4, "absences": 2}'
```

Interpret this as a **model scenario**, not as a causal guarantee that changing a feature will cause a particular outcome.

## Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard can:
- enter a student profile
- run the performance model
- show model prediction
- show SHAP-based explanations when the model is available
- run a what-if scenario
- display placement-model readiness once a placement model is trained

## Academic integrity / results policy

Do **not** copy any example metrics from external notebooks. Run the training scripts on the dataset you actually use and put those measured values into your report/presentation. Published-study values and your own measured values must remain clearly separated.

## Recommended project evolution

1. UCI real benchmark → leakage-safe performance prediction.
2. Permissioned institutional placement data → placement prediction.
3. SHAP → student-specific explanation.
4. Counterfactual/what-if → controllable improvement scenarios.
5. Role skill-gap engine → target-role analysis.
6. Grounded GenAI → 30/60/90-day action plans based only on verified model outputs.
7. Closed-loop re-evaluation → retrain/re-score after new assessments.

## Important limitation

The UCI Student Performance dataset is real/public but is from Portuguese secondary schools, not an Indian engineering college. It is appropriate as a **research benchmark**, not as proof of Indian placement behavior. Likewise, the public placement benchmark should not be described as institutionally collected real student data without separate provenance evidence.
