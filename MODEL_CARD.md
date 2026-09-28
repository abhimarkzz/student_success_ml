# ML Model Card

## Model family
**CatBoost Gradient Boosted Decision Trees**.

## Intended use
Early-warning student performance modeling and placement-outcome modeling for educational analytics research. Outputs are decision-support signals, not automatic academic or hiring decisions.

## Performance model target
UCI Student Performance `G3` (final grade, 0–20).

Main feature policy excludes `G1`, `G2`, `G3`, and `fail` so the model can be interpreted as an early-warning system.

## Placement model target
A binary placement-status field such as `PlacementStatus`.

## Why this architecture
- Native categorical feature handling is useful for student attributes such as school, guardian, family support, activities and similar fields.
- Gradient boosting handles nonlinear interactions among academic and behavioral variables.
- Tree-based feature attribution supports SHAP explanations.
- CatBoost models can run locally on CPU for a B.Tech project and can optionally use GPU where available.

## Not a causal model
A feature with high importance is not automatically a cause. What-if simulations show how the **model prediction** changes under an input scenario; they do not prove that changing the real-world feature will cause the predicted change.

## Fairness and privacy
Do not use sensitive personal data without legitimate educational purpose and permission. De-identify institutional data. Audit performance across relevant groups before any operational use.

## Dataset limitations
The UCI performance benchmark is from two Portuguese secondary schools. A placement model trained on a public benchmark should not be presented as representative of an Indian college population unless validated on appropriately sourced Indian/institutional data.
