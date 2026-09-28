# Project notes for the B.Tech presentation

## Defensible novelty claim
Do not claim that CatBoost, SHAP, counterfactuals, or GenAI individually are novel. Recent literature already uses these elements.

The defensible contribution is the **integration into one closed-loop student-success workflow**:

**Predict → Explain → Identify Skill Gap → Simulate What-If → Generate Action Plan → Re-evaluate**

## Current ML implementation boundary
This package implements:
- UCI real/public performance benchmark
- CatBoost regression for final grade
- CatBoost binary pass-risk classification
- SHAP local explanation
- Model-based what-if simulation
- CatBoost placement classifier adapter

GenAI, role/job semantic matching and closed-loop longitudinal re-evaluation are the next integration layer.
