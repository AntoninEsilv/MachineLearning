# Contributions (ownership matrix)

Each module has one owner, who must be able to explain every line of it at the oral defense.
Rows marked *to confirm* were proposed from the work plan of step 1 and must be validated by their owner.

| Module | Files | Owner | Status |
|---|---|---|---|
| 1. Data exploration and cleaning | `notebooks/01_data_exploration.ipynb`, `dataset/processed/bank_clean.csv` | Baudouin Parent | done |
| 2. Features, split and preprocessing | `code/step2_split.py`, `code/step2_preprocessing.py`, `dataset/train.csv`, `dataset/test.csv` | Vivien Perrault | done (Enki fixed the import name and added the grouping of rare categories, `min_frequency=20`, announced in notebook 01) |
| 3. Advanced models (v0.2) | `notebooks/03_advanced_models.ipynb`, `random_forest` and `lightgbm` in `src/models.py` | Baudouin Parent (Antonin Charron *to confirm*) | done (random forest, LightGBM, tuned with the shared folds) |
| 3c. Ensemble (v0.3) and model outside the course (v0.4) | `notebooks/06_ensemble_and_mlp.ipynb`, `stacking`, `soft_voting` and `mlp` in `src/models.py` | Enki Millet (to hand over if another member defends it) | done (stacking chosen in CV, multilayer perceptron with references) |
| 3b. Feature selection | `notebooks/05_feature_selection.ipynb` | Vivien Perrault | done (no method beats the full feature set in a reliable way) |
| 4. Baseline, AutoML v0.1 and evaluation protocol | `notebooks/02_baseline_automl.ipynb`, `notebooks/04_model_comparison.ipynb`, `src/common.py`, `src/evaluation.py`, `src/models.py` | Enki Millet | done; notebook 04 run with every step 3 model (test, paired bootstrap, OOF threshold check, temporal split, final conclusion) |

Shared: report, slides, `ai_governance/` (each member writes at least one entry in `PROMPT_LOG.md` and one in `AI_AUDIT.md`).
