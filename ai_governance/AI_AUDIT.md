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
  (it is a random choice). The decision tree had 0.129 for the same reason.
* **Fix (v2):** for the tied clients at the cut, we count the expected number of yes (as if they were picked at
  random). Checked with small hand-made cases (constant score gives 0.2, a tie of 3 with 1 yes gives 1/3).
  The gain curve now breaks ties at random with a fixed seed.
* **Result:** dummy 0.052 → 0.200 and decision tree 0.129 → 0.222 on the temporal split; no change for the
  models without ties (logistic regression 0.402).

### E2. Hyperparameter chosen on noise, at the edge of the grid

* **AI output (v1 of the baseline in notebook 02):** grid search on `C` in {0.001, ..., 10} and keep
  `gs.best_params_`, which gave `C = 10`.
* **Problem:** `C = 10` was the last value of the grid, so we did not know if a larger value would be better,
  and the CV PR-AUC was flat (0.4464 at `C = 0.01`, 0.4501 at `C = 10`) while the standard deviation between
  folds is about 0.013. Taking the maximum was choosing on noise, and choosing the weakest regularisation,
  although `euribor3m`, `emp.var.rate` and `nr.employed` are collinear (VIF > 30).
* **Fix (v2):** grid extended to `C = 100` (score 0.4500, so the plateau is confirmed) and the
  **one standard error rule**: among the settings within one standard error (std / sqrt(5)) of the best score,
  keep the most regularised one. Result: `C = 0.01`, `class_weight=None`.
* **Result:** CV PR-AUC 0.4464 instead of 0.4501 (a difference 3 times smaller than the fold std), a simpler
  and more stable model, and coefficients of more reasonable size (the economic coefficient went from
  -2.36 to -0.43).
