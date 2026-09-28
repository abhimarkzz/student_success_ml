#!/usr/bin/env bash
set -euo pipefail

python scripts/download_datasets.py --uci
python scripts/train_performance.py --subject mat --tune-trials 20
python scripts/explain_student.py --model models/performance_mat_regressor.cbm --input examples/student_mat_example.csv
python scripts/what_if.py --model models/performance_mat_regressor.cbm --input examples/student_mat_example.csv --changes '{"studytime": 4, "absences": 2}'

echo "Performance model complete."
echo "For placement:"
echo "  python scripts/download_datasets.py --placement-benchmark"
echo "  python scripts/train_placement.py --csv data/raw/placement/train.csv"
echo "Then run: streamlit run app/streamlit_app.py"
