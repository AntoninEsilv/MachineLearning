# Bank telemarketing: who will subscribe to a term deposit?

Machine Learning project 2026, DIA2. Baudouin Parent, Vivien Perrault, Antonin Charron, Enki Millet.

Binary classification on the UCI Bank Marketing dataset (`bank-additional-full.csv`, 41,188 calls, 11 % of yes).
The goal is to rank the clients so the bank calls the most likely ones first. Main metric: PR-AUC.
Business goal: reach at least 60 % of the subscribers by calling 20 % of the clients.

## Structure
```
dataset/raw/                 original data (UCI, CC BY 4.0)
dataset/processed/           bank_clean.csv, made by notebook 01
dataset/train.csv, test.csv  stratified 80/20 split, made by code/step2_split.py
code/                        step2_split.py, step2_preprocessing.py (split and preprocessor)
notebooks/
  01_data_exploration.ipynb  quality, statistical tests, cleaning           (step 2.1)
  02_baseline_automl.ipynb   split plan, dummy, logistic regression baseline,
                             benchmark, AutoML v0.1 with standard models     (step 2.2-2.5)
  03_advanced_models.ipynb   random forest, boosting, ensemble, model outside the course (step 3)
  04_model_comparison.ipynb  test set, confidence intervals, comparison with the baseline, temporal test (step 3)
  05_feature_selection.ipynb compare feature-selection methods with CV and a final test-set check
src/
  common.py                  loads the split, CV folds, temporal split (uses code/step2_preprocessing.py)
  evaluation.py              metrics, recall in the top 20 %, bootstrap
  models.py                  registry of the final models compared in notebook 04
results/                     csv tables and tuned hyperparameters (best_params_*.json)
report/figures/              figures for the report
ai_governance/               CONTRIBUTIONS.md, PROMPT_LOG.md, AI_AUDIT.md
rendus/                      deliverables already sent (step 1)
```

## Reproduce
```
pip install -r requirements.txt
python code/step2_split.py      # only if dataset/train.csv and test.csv are missing
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_data_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_baseline_automl.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_model_comparison.ipynb
```
All paths are relative, all seeds are 42.

## Adding a step 3 model
1. Tune it in notebook 03 with `cv_folds()`, `make_preprocessor()` and `SCORING` from `src/` (same folds, same score).
2. Save the tuned hyperparameters with `save_best_params("my_model", params)`.
3. Add a builder for it in `BUILDERS` in `src/models.py`.
4. Re-run notebook 04: the model is evaluated on the test set and on the temporal split automatically.

## Results so far (test set, 8,236 calls)
| Model | PR-AUC [95 % CI] | ROC-AUC | Subscribers reached with 20 % of calls |
|---|---|---|---|
| Dummy | 0.113 | 0.500 | 20.0 % |
| Logistic regression (baseline) | 0.454 [0.418, 0.488] | 0.800 | 64.0 % |
| Best AutoML v0.1 model on test (decision tree) | 0.472 [0.434, 0.503] | 0.805 | 64.0 % |

The gain of the tree is not reliable (lower than the baseline in cross-validation, collapses on the temporal split).
