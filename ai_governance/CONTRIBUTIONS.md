# Contributions (ownership matrix)

The pipeline is split into 4 modules, one owner per student. The owner defends the module at the oral and must be
able to explain every line of it.

| Module | Files | Owner | Status |
|---|---|---|---|
| 1. Data exploration and cleaning | `notebooks/01_data_exploration.ipynb`, `dataset/processed/bank_clean.csv` | Baudouin Parent | done |
| 2. Features, split, preprocessing and feature selection | `code/step2_split.py`, `code/step2_preprocessing.py`, `dataset/train.csv`, `dataset/test.csv`, `notebooks/05_feature_selection.ipynb` | Vivien Perrault | done |
| 3. Advanced models (v0.2), ensemble (v0.3) and model outside the course (v0.4) | `notebooks/03_advanced_models.ipynb`, `notebooks/06_ensemble_and_mlp.ipynb`, `random_forest`, `lightgbm`, `stacking`, `soft_voting` and `mlp` in `src/models.py` | Antonin Charron | done |
| 4. Baseline, AutoML v0.1 and evaluation protocol | `notebooks/02_baseline_automl.ipynb`, `notebooks/04_model_comparison.ipynb`, `src/common.py`, `src/evaluation.py`, `src/models.py` | Enki Millet | done |

Shared: report, slides, `ai_governance/` (each member writes at least one entry in `PROMPT_LOG.md` and one in
`AI_AUDIT.md`).
