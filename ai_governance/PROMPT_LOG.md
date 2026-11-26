# Prompt log: high-stakes AI interactions

4 to 6 entries in total, at least 1 per member. For each: the tool, the exact prompt, the constraints given
(data schema, leakage boundaries, assumptions to check, metric), what came back and what we kept.

---

## Baudouin Parent (module 1)

*To write.*

---

## Vivien Perrault (module 2)

*To write.*

---

## Antonin Charron (module 3)

*To write.*

---

## Enki Millet (module 4)

### P1. Baseline, AutoML v0.1 and evaluation protocol
* **Tool:** Claude Code (Claude Opus 5.5), working directly in the repository, 26/11/2026.
* **Prompt 1 (bare, French, exact):** "regarde ce que je dois faire, en sachant qu'on doit avoir fini le step 3 et
  qu'on doit preparer un oral. [...] les steps que je dois faire"
  → The AI read the guidelines and the repo and listed the missing work; it pointed out that the notebook 01
  path (`data/` instead of `dataset/`) was broken and that `row_order` must not be used as a feature.
* **Prompt 2 (exact):** "Fais tout ce que tu peux faire que je dois faire en m'expliquant ce que tu as fait [...]
  Assure toi que la step 1 et step 2.1 sont carrés."
* **Weakness of these prompts:** they do not state the schema, the metric or the leakage rules; the AI had to take
  them from the step 1 document and the guidelines. This is the "bare prompt" case of the brief, so we checked
  every point by hand afterwards (see `AI_AUDIT.md` E1 and E2).
* **Checks done on the output:** encoder and scaler inside the `Pipeline` (fitted on training folds only),
  `duration` and `row_order` absent from the features (assert / print in notebook 02), same test rows in 02 and
  04 (`results/test_rows.csv` + assert), threshold chosen on out-of-fold train scores, test set used once.

### P2. Leakage audit of the pipeline (constrained prompt)
* **Tool:** Claude Code, 26/11/2026. Compared to P1, the prompt gives the data schema, the leakage boundaries
  and asks for a proof per point.
* **Prompt (exact):**
  > Audit `notebooks/02_baseline_automl.ipynb` and `src/` for data leakage. Schema: 41,176 rows, target `y` (0/1,
  > 11.3 % positives), features = `FEATURES` in `code/step2_preprocessing.py`; `duration` and `row_order` must NOT
  > be features. Check: (1) OneHotEncoder and StandardScaler are fitted only on training folds (inside the
  > Pipeline); (2) the test set (`dataset/test.csv`) is never used in 02 for tuning or threshold choice; (3) the
  > threshold in 04 is chosen on out-of-fold train scores only; (4) the deduplication before the split cannot leak.
  > For each point answer PASS/FAIL with the line of code as proof. Do not change code without telling me why.
* **Answer (summary):**
  1. PASS: every model in `src/models.py` is `Pipeline([("prep", ...), ("model", ...)])`, and all the CV calls
     (`GridSearchCV`, `RandomizedSearchCV`, `cross_validate`, `cross_val_predict`, `learning_curve`) receive the
     whole pipeline. `common.py` asserts that `row_order` and `duration` are not in `FEATURES`.
  2. PASS: in notebook 02 the test set is only used to print its size and share of yes and to save its row ids
     (`results/test_rows.csv`).
  3. PASS: in notebook 04, `best_f1_threshold(y_train, oof)` uses `cross_val_predict` on the train set.
  4. PASS, with an extra check: removing exact duplicates before the split only removes copies, it cannot carry
     information from test to train. But after removing `duration`, **7.6 % of the test rows (626) have exactly
     the same features as a train row** (different calls of similar clients). We measured whether this inflates the
     scores: without these rows the PR-AUC goes *up* (baseline 0.454 → 0.459, tree 0.472 → 0.476, KNN 0.448 →
     0.455), so the twins do not help the models; they are typical clients never contacted before (9.7 % of yes).
  * Extra risk found: `emp.var.rate` and `nr.employed` are quarterly indicators, maybe published after the call
    (look-ahead). Not fixable with this data; added to the risks in notebook 02.
* **Kept:** no code change for points 1-4; the risk was added to notebook 02, section 6.

### P3. Statistical choices of the evaluation (constrained prompt)
* **Tool:** Claude Code, 26/11/2026, same conditions as P2.
* **Prompt (exact):**
  > Check the statistical choices of notebook 04. Main metric is PR-AUC because of the 11 % imbalance; do not use
  > accuracy. Verify: (1) the bootstrap CI resamples test rows with replacement and the paired bootstrap uses the
  > same indices for both models; (2) the one-standard-error rule in notebook 02 uses std/sqrt(5) correctly;
  > (3) PR-AUC values on the temporal split are not compared directly to the random split (different prevalence:
  > 0.308 vs 0.113). If an assumption is violated, propose a fallback (e.g. a ranking metric instead of a
  > thresholded one). Answer point by point with numbers from the outputs.
* **Answer (summary):**
  1. PASS: `rng.integers(0, n, n)` = resampling with replacement; `paired_bootstrap_diff` uses the same `idx` for
     both models. Limit: the CI only covers the randomness of the test sample, not of the training (the model is
     fitted once).
  2. **FAIL (small):** scikit-learn's `std_test_score` is `np.std` with `ddof=0` (population std, 0.00558 for
     `C = 0.1`), while a standard error needs the sample std (`ddof=1`, 0.00624). The SE was 11 % too small.
     Fixed (see `AI_AUDIT.md` E3); the chosen `C` stays 0.1.
  3. PASS: notebook 04 divides PR-AUC by the share of yes and also shows the ROC-AUC, which does not depend on
     it. This ratio is only a rough correction, so the ROC-AUC is the fallback metric for this comparison (text
     updated). Threshold metrics (precision, F1) are not reported on the temporal split, since a threshold chosen
     with 11 % of yes is meaningless with 31 %.
* **Kept:** the ddof fix in notebook 02, one sentence added in notebook 04.
