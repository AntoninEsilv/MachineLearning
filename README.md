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
  03_advanced_models.ipynb   step 3 part 1 (v0.2): tuned random forest and LightGBM
  04_model_comparison.ipynb  test set, confidence intervals, comparison with the baseline, temporal test (step 3)
  05_feature_selection.ipynb compare feature-selection methods with CV and a final test-set check
  06_ensemble_and_mlp.ipynb  step 3 part 2: limitations of v0.1/v0.2, ensemble (v0.3, stacking vs
                             soft voting), model outside the course (v0.4, multilayer perceptron)
src/
  common.py                  loads the split, CV folds, temporal split (uses code/step2_preprocessing.py)
  evaluation.py              metrics, recall in the top 20 %, bootstrap
  models.py                  registry of the final models compared in notebook 04
results/                     csv tables and tuned hyperparameters (best_params_*.json)
report/                      Report_Group_XX.pdf (final report) and its LaTeX source
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
jupyter nbconvert --to notebook --execute --inplace 03_advanced_models.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_ensemble_and_mlp.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_model_comparison.ipynb
```
All paths are relative, all seeds are 42.

## Adding a step 3 model
1. Tune it in notebook 03 or 06 with `cv_folds()`, `make_preprocessor()` and `SCORING` from `src/` (same folds, same score).
2. Save the tuned hyperparameters with `save_best_params("my_model", params)`.
3. Add a builder for it in `BUILDERS` in `src/models.py`.
4. Re-run notebook 04: the model is evaluated on the test set and on the temporal split automatically.

## Results (test set, 8,236 calls)
| Version | Model | CV PR-AUC | Test PR-AUC [95 % CI] | Test ROC-AUC | Subscribers reached with 20 % of calls | Temporal split ROC-AUC |
|---|---|---|---|---|---|---|
| | Dummy | 0.113 | 0.113 | 0.500 | 20.0 % | 0.500 |
| v0.1 | Logistic regression (baseline) | 0.448 | 0.454 [0.418, 0.488] | 0.800 | 64.0 % | **0.748** |
| v0.1 | Decision tree (best v0.1 model on test) | 0.439 | 0.472 [0.434, 0.503] | 0.805 | 64.0 % | 0.542 |
| v0.2 | Random forest | 0.465 | **0.498** [0.461, 0.530] | 0.817 | 67.1 % | 0.679 |
| v0.2 | LightGBM | 0.467 | 0.490 [0.453, 0.523] | 0.814 | 66.6 % | 0.676 |
| v0.3 | Stacking (logistic regression + random forest + LightGBM) | **0.470** | 0.495 [0.458, 0.527] | 0.817 | 66.9 % | 0.715 |
| v0.4 | Multilayer perceptron (outside the course) | 0.453 | 0.467 [0.432, 0.502] | 0.804 | 65.2 % | 0.594 |

* Random forest, stacking and LightGBM beat the baseline in a reliable way (paired bootstrap: PR-AUC +0.043,
  +0.040 and +0.036, recall at 20 % +2.8, +2.6 and +2.5 points, CIs above 0), in the same direction as
  cross-validation. They cannot be separated from each other.
* The MLP is not a reliable gain (+0.013 [-0.001, +0.026]), like in cross-validation. The gain of the decision
  tree is not reliable either (lower than the baseline in cross-validation).
* Temporal split (train on the oldest 80 %, test on the most recent 20 %): every flexible model falls below the
  logistic regression, because `euribor3m` is out of the training range for 99 % of the recent calls. The
  stacking resists best (it contains the logistic regression), the MLP collapses.
* Details: `notebooks/06_ensemble_and_mlp.ipynb` (cross-validation of v0.3 and v0.4) and
  `notebooks/04_model_comparison.ipynb` (test set, paired bootstrap, threshold, temporal split, conclusion).

## References
* Moro, S., Cortez, P., Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing.
  *Decision Support Systems*, 62, 22-31.
* Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241-259.
* Kadra, A., Lindauer, M., Hutter, F., Grabocka, J. (2021). Well-tuned simple nets excel on tabular datasets.
  *Advances in Neural Information Processing Systems 34* (NeurIPS 2021).
* Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on
  typical tabular data? *NeurIPS 2022, Datasets and Benchmarks track*.
