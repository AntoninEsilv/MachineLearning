# Contributions (ownership matrix)

The pipeline is split into 4 modules, one owner per student. The owner defends the module at the oral and must be
able to explain every line of it. When a part was implemented by another member, it is written in the last column.
AI use for each module is in `PROMPT_LOG.md` and `AI_AUDIT.md`.

| Module | Files | Owner | Implemented by | Status |
|---|---|---|---|---|
| 1. Data exploration and cleaning | `notebooks/01_data_exploration.ipynb`, `dataset/processed/bank_clean.csv` | Baudouin Parent | Baudouin Parent | done |
| 2. Features, split, preprocessing and feature selection | `code/step2_split.py`, `code/step2_preprocessing.py`, `dataset/train.csv`, `dataset/test.csv`, `notebooks/05_feature_selection.ipynb` | Vivien Perrault | Vivien Perrault (Enki Millet fixed the import name and added the grouping of rare categories, `min_frequency=20`, announced in notebook 01) | done |
| 3. Advanced models (v0.2), ensemble (v0.3) and model outside the course (v0.4) | `notebooks/03_advanced_models.ipynb`, `notebooks/06_ensemble_and_mlp.ipynb`, `random_forest`, `lightgbm`, `stacking`, `soft_voting` and `mlp` in `src/models.py` | Antonin Charron *(to confirm)* | notebook 03: Baudouin Parent; notebook 06: Enki Millet with Claude Code (`PROMPT_LOG.md` P4) | done |
| 4. Baseline, AutoML v0.1 and evaluation protocol | `notebooks/02_baseline_automl.ipynb`, `notebooks/04_model_comparison.ipynb`, `src/common.py`, `src/evaluation.py`, `src/models.py` | Enki Millet | Enki Millet | done (notebook 04 run with every step 3 model) |

Shared: report, slides, `ai_governance/` (each member writes at least one entry in `PROMPT_LOG.md` and one in
`AI_AUDIT.md`).
