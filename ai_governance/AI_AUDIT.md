# AI audit: flaws caught in AI outputs and how we fixed them

At least 4 entries in total, at least 1 per member. Each entry: what the AI produced, why it is wrong,
how we found it, the fix, and where the fixed version is.

---

## Baudouin Parent (module 1)

### B1. Chi-square test used without checking its condition

* **AI output (first version of notebook 01):** a function running a chi-square test for
  each categorical variable, with the comment "assumption checked: expected count ≥ 5". The minimum expected
  count was displayed but never checked.
* **Problem:** the chi-square test is only valid if every cell has an expected count of at least 5. It was
  0.34 for `default` (only 3 rows with `yes`) and 2.03 for `education` (18 rows with `illiterate`), so the
  p-values of these two variables were not reliable.
* **How we caught it:** by reading the `min_expected` column of the output table.
* **Fix:** categories with fewer than 20 rows are grouped into `other`, and an `assert` checks the condition.
  The assert failed: each variable has a single rare category, so `other` was still too small. Final fix: when
  `other` is still below the condition, its rows are left out of the test (3 and 18 rows out of 41,176).
* **Result:** the assert passes and the conclusions do not change (`default` p < 0.001, Cramér's V ≈ 0.10;
  `education` p < 0.001, V ≈ 0.07).

## Vivien Perrault (module 2)

*To write.*

---

## Antonin Charron (module 3)

### A1. MLP early stopping based on accuracy instead of PR-AUC

* **AI output (first version of the MLP):** `MLPClassifier(early_stopping=True, random_state=42)` to avoid
  overfitting during training.
* **Problem:** in scikit-learn, `early_stopping=True` keeps the epoch with the best **accuracy** on an internal
  validation split. Because our dataset has only 11.3 % of `yes`, predicting `no` for everyone already gives
  88.7 % accuracy, so the accuracy stays flat and the network stops at a random epoch.
* **How we caught it:** by checking why the MLP score was unstable and reading the scikit-learn documentation of
  `MLPClassifier`.
* **Fix:** set `early_stopping=False` in `src/models.py` and tune the number of epochs (`max_iter` from 10 to 150)
  and the L2 penalty (`alpha`) directly in 5-fold cross-validation using PR-AUC (`notebooks/06_ensemble_and_mlp.ipynb`).
* **Result:** the tuned MLP (`max_iter = 80`, `alpha = 0.01`, layers `[128, 64]`) reaches a CV PR-AUC of 0.453
  instead of 0.368 with the default settings, and reduces the train-CV gap from 0.37 to 0.08.

---

## Enki Millet (module 4)

### E1. Recall in the top 20 % judged on the row order (= time order) when scores are tied

* **AI output (v1 of `recall_at_top` in `src/evaluation.py`):** sort the clients by score with
  `np.argsort(-score, kind="stable")` and take the first 20 %.
* **Problem:** when several clients have the same score, the stable sort keeps them in the original row order.
  In this dataset the row order is the time order, and the share of yes grows a lot over time. So for models
  that give many equal scores (dummy, decision tree), the metric depended on *when* the tied calls were made,
  not on the model.
* **How we caught it:** on the temporal split (notebook 04, section 4), the dummy model got a recall of
  **0.052** in the top 20 %. A model that gives the same score to everybody should get exactly 0.20 on average
  (it is a random choice). The decision tree was also too low (0.129) for the same reason.
* **Fix (v2):** for the tied clients at the cut, we count the expected number of yes (as if they were picked at
  random). Checked with small hand-made cases (constant score gives 0.2, a tie of 3 with 1 yes gives 1/3).
  The gain curve now breaks ties at random with a fixed seed.
* **Result:** dummy 0.052 → 0.200 on the temporal split, the decision tree is now judged on its scores only;
  no change for the models without ties (logistic regression, linear SVM).

### E2. Hyperparameter chosen on noise, at the edge of the grid

* **AI output (v1 of the baseline in notebook 02):** grid search on `C` in {0.001, ..., 10} and keep
  `gs.best_params_`, which gave `C = 10`.
* **Problem:** `C = 10` was the last value of the grid, so we did not know if a larger value would be better.
  When we extended the grid to `C = 100` (and moved to the shared split of `code/step2_split.py`), the "best"
  value became `C = 100`, again the edge: the CV PR-AUC is flat (0.4484 at `C = 0.1`, 0.4497 at `C = 100`)
  while the standard deviation between folds is about 0.006. Taking the maximum was choosing on noise, and
  choosing the weakest regularisation, although `euribor3m`, `emp.var.rate` and `nr.employed` are collinear
  (VIF > 30).
* **Fix (v2):** **one standard error rule**: among the settings within one standard error (sample std / sqrt(5)) of the
  best score, keep the most regularised one. Result: `C = 0.1`, `class_weight=None` (notebook 02, section 3).
* **Result:** CV PR-AUC 0.4484 instead of 0.4497 (a difference smaller than the fold std), a simpler and more
  stable model, chosen by a rule decided before looking at the test set.

### E3. Standard error computed with the population std

* **AI output (v2 of the one standard error rule, notebook 02):**
  `se = res.loc[best_i, "std_test_pr_auc"] / np.sqrt(cv.get_n_splits())`.
* **Problem:** scikit-learn computes `std_test_*` with `np.std`, i.e. `ddof=0`. A standard error of the mean needs
  the sample standard deviation (`ddof=1`). With 5 folds the SE was too small by a factor sqrt(4/5) (-11 %),
  so the rule was slightly too strict.
* **How we caught it:** constrained audit prompt P3; checked by recomputing the 5 fold scores of `C = 0.1`:
  `std_test_score` = 0.00558 = `np.std(ddof=0)`, while `np.std(ddof=1)` = 0.00624.
* **Fix (v3):** `std_sample = std * sqrt(k / (k - 1))`, then `se = std_sample / sqrt(k)`.
* **Result:** the threshold goes from 0.4471 to 0.4468; the chosen model does not change (`C = 0.1`).
  Remaining limit: the 5 fold scores are not independent (the training folds overlap), so the SE is still an
  approximation; the rule is a heuristic, not a test.
