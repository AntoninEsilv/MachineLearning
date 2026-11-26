# Contributions (ownership matrix)

Each module has one owner, who must be able to explain every line of it at the oral defense.
Rows marked *to confirm* were proposed from the work plan of step 1 and must be validated by their owner.

| Module | Files | Owner | Status |
|---|---|---|---|
| 1. Data exploration and cleaning | `notebooks/01_data_exploration.ipynb`, `dataset/processed/bank_clean.csv` | Baudouin Parent | done |
| 2. Features, split and preprocessing | `code/step2_split.py`, `code/step2_preprocessing.py`, `dataset/train.csv`, `dataset/test.csv` | Vivien Perrault | done (Enki fixed the import name and added the grouping of rare categories, `min_frequency=20`, announced in notebook 01) |
| 3. Advanced models, ensemble, model outside the course | `notebooks/03_advanced_models.ipynb`, entries in `src/models.py` | Antonin Charron *(to confirm)*; part 1 committed by Baudouin Parent | part 1 done (random forest, LightGBM, tuned with the shared folds); ensemble and model outside the course: to do |
| 3b. Feature selection | `notebooks/05_feature_selection.ipynb` | Vivien Perrault | done (no method beats the full feature set in a reliable way) |
| 4. Baseline, AutoML v0.1 and evaluation protocol | `notebooks/02_baseline_automl.ipynb`, `notebooks/04_model_comparison.ipynb`, `src/common.py`, `src/evaluation.py`, `src/models.py` | Enki Millet | done for the baseline and v0.1; notebook 04 re-run with the random forest and LightGBM (test, paired bootstrap, OOF threshold check, temporal split). To re-run once more when the ensemble and the model outside the course are added |

Shared: report, slides, `ai_governance/` (each member writes at least one entry in `PROMPT_LOG.md` and one in `AI_AUDIT.md`).
