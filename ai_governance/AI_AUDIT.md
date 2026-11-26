# AI audit: flaws caught in AI outputs and how we fixed them

At least 4 entries in total, at least 1 per member. Each entry: what the AI produced, why it is wrong,
how we found it, the fix, and where the fixed version is.

---

## Baudouin Parent (module 1)

*To write.* Notebook 01 already refers to an entry about the chi-square condition (expected counts < 5 for
`default` and `education`, fixed by grouping rare categories and excluding the rest from the test).

---

## Vivien Perrault (module 2)

*To write.*

---

## Antonin Charron (module 3)

*To write.*

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
