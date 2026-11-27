# Prompt log: high-stakes AI interactions

5 entries in total (4 to 6 required by the brief, 3 to 5 by the guidelines), at least 1 per member. For each: the tool, the exact prompt, the constraints given
(data schema, leakage boundaries, assumptions to check, metric), what came back and what we kept.

---

## Baudouin Parent (module 1)

### B1. Bare prompt for step 3 models, used as a control
* **Tool:** Gemini.
* **Prompt (bare, exact):**
  > I have a bank marketing dataset (bank-cleaned.csv,Schema: 41,176 rows, target y = yes/no, 11 % of yes).
  > Write Python code to train a random forest and a LightGBM model, tune them, prevent leakage.
* **Constraints given:** the dataset schema, an explicit instruction to prevent data leakage, the general pipeline
  and the models to train.
* **What came back:** a complete script with a stratified 80/20 split, preprocessing inside a Pipeline,
  `RandomizedSearchCV` with `scoring="average_precision"`. There was no data leakage (we had already removed `duration`).
* **What we did not keep:** class weights combined with a fixed 0.5 threshold applied on the test set,
  `pdays = 999` scaled as a number, duplicates kept, no baseline and no confidence interval.
* **Lesson:** without an explicit protocol in the prompt, the AI follows generic good practice but misses
  the decisions specific to our data (sentinel values, threshold chosen out-of-fold, comparison with the
  baseline). These must be written into the prompt as constraints, as in Enki's P2.

## Vivien Perrault (module 2)

### V1. Feature-selection implementation
* **Tool:** GitHub Copilot coding agent, working directly in the repository.
* **Prompt (initial, exact):**
  > "Can you help me implement test feature selection?"
* **Constraints given:** this initial prompt was broad; it did not explicitly specify the data split, the leakage
  boundaries, the metric, or the cross-validation protocol. The leakage constraint, preprocessing and
  selection pipeline, five-fold stratified CV, and PR-AUC protocol were derived from the project and notebook
  conventions rather than stated in this prompt.
* **What came back:** a feature-selection implementation comparing all features, mutual information at 50 %
  and 75 %, and L1 median and stricter thresholds. Preprocessing and selection were placed inside the same
  Pipeline, evaluated with the project's five-fold stratified CV and PR-AUC. L1 median was the CV winner
  (about 0.4489 PR-AUC, versus 0.4484 for all features).
* **What we kept or changed:** kept the CV comparison and the in-pipeline preprocessing/selection. Rejected the
  first script design that evaluated every candidate on the held-out test set, because that reused test data to
  assess model-selection alternatives. The fixed version in notebooks/05_feature_selection.ipynb uses test
  data once for the CV-selected method and the all-feature baseline only.
* **Follow-up prompt (exact):**
  > "Plutôt faire un notebook à la place du code step3_features_selection"

  It produced `notebooks/05_feature_selection.ipynb`, which keeps the CV protocol and compares only the CV-selected
  L1 median method with the all-feature baseline on the test set.
* **Lesson:** a broad implementation prompt is not sufficient to encode a valid evaluation protocol; the
  repository conventions must be explicit or reviewed by hand. Test-set results must not be used to choose the
  final method. Changing the output format (script to notebook) was also the occasion to correct the evaluation
  design.


---

## Antonin Charron (module 3)

### A1. Prompt for the ensemble (v0.3) and the MLP (v0.4)
* **Tool:** Claude.
* **Prompt (exact):**
  > Using our existing pipeline (`make_preprocessor()`, 5-fold stratified CV `cv_folds()`, metric PR-AUC),
  > write Python code to compare a Soft Voting and a Stacking ensemble combining our tuned logistic regression,
  > random forest and LightGBM, and train a tuned MLPClassifier as a model outside the course.
* **Constraints given:** We gave the base models to combine, the preprocessing pipeline, the 5-fold cross-validation
  and the main metric (PR-AUC on `train.csv` only).
* **What came back:** code for `VotingClassifier(voting="soft")`, `StackingClassifier` with a logistic regression
  meta-model, and a `RandomizedSearchCV` for `MLPClassifier`.
* **What we did not keep:** `StackingClassifier` with default `cv=5` instead of our shared `cv_folds()`, and
  `early_stopping=True` on the `MLPClassifier` (which monitors accuracy instead of PR-AUC).
* **Lesson:** even when we give the metric in the prompt, the AI uses default scikit-learn options
  (`cv=5`, `early_stopping=True`) that do not match an imbalanced dataset (11 % of yes).

---

## Enki Millet (module 4)

### P1. Bare prompts for the implementation (baseline, protocol, step 3 models)
* **Tool:** Claude Code (Claude Opus 5.5), working directly in the repository, 26/11/2026.
* **Prompts (bare, French, exact):**
  1. "regarde ce que je dois faire, en sachant qu'on doit avoir fini le step 3 et qu'on doit preparer un oral.
     [...] les steps que je dois faire" → the AI read the guidelines and the repo and listed the missing work; it
     pointed out that the notebook 01 path (`data/` instead of `dataset/`) was broken and that `row_order` must not
     be used as a feature.
  2. "Fais tout ce que tu peux faire que je dois faire en m'expliquant ce que tu as fait [...] Assure toi que la
     step 1 et step 2.1 sont carrés."
  3. Later, for step 3 (re-run of notebook 04 with the new models, then notebook 06: stacking and MLP): "fais ce
     qu'il reste à faire pour que la step 3 sois entierement fini."
* **Weakness of these prompts:** they do not state the schema, the metric or the leakage rules; the AI had to take
  them from the step 1 document, the guidelines and the repo. This is the "bare prompt" case of the brief, so we
  checked every point by hand afterwards (see `AI_AUDIT.md` E1 and E2, and P2 for the constrained versions).
* **Checks done on the output:** encoder and scaler inside the `Pipeline` (fitted on training folds only),
  `duration` and `row_order` absent from the features (assert / print in notebook 02), same test rows in 02 and
  04 (`results/test_rows.csv` + assert), threshold chosen on out-of-fold train scores, test set used once.
  For step 3: the meta-model of the stacking learns on out-of-fold scores (`cv=cv_folds()`); `early_stopping` of
  the MLP disabled because scikit-learn monitors the accuracy (almost flat with 11 % of yes); every model kept is
  chosen on CV, never on the test set; the 4 references checked to exist; the whole project re-run from a clean
  clone with the same results.

### P2. Constrained audit prompts: data leakage, then statistical choices
* **Tool:** Claude Code, 26/11/2026, two prompts in a row. Compared to P1, they give the data schema, the leakage
  boundaries, the assumptions to check and a fallback, and ask for a proof per point.
* **Prompt 1, leakage (exact):**
  > Audit `notebooks/02_baseline_automl.ipynb` and `src/` for data leakage. Schema: 41,176 rows, target `y` (0/1,
  > 11.3 % positives), features = `FEATURES` in `code/step2_preprocessing.py`; `duration` and `row_order` must NOT
  > be features. Check: (1) OneHotEncoder and StandardScaler are fitted only on training folds (inside the
  > Pipeline); (2) the test set (`dataset/test.csv`) is never used in 02 for tuning or threshold choice; (3) the
  > threshold in 04 is chosen on out-of-fold train scores only; (4) the deduplication before the split cannot leak.
  > For each point answer PASS/FAIL with the line of code as proof. Do not change code without telling me why.
* **Answer to prompt 1 (summary):**
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

* **Prompt 2, statistical choices (exact):**
  > Check the statistical choices of notebook 04. Main metric is PR-AUC because of the 11 % imbalance; do not use
  > accuracy. Verify: (1) the bootstrap CI resamples test rows with replacement and the paired bootstrap uses the
  > same indices for both models; (2) the one-standard-error rule in notebook 02 uses std/sqrt(5) correctly;
  > (3) PR-AUC values on the temporal split are not compared directly to the random split (different prevalence:
  > 0.308 vs 0.113). If an assumption is violated, propose a fallback (e.g. a ranking metric instead of a
  > thresholded one). Answer point by point with numbers from the outputs.
* **Answer to prompt 2 (summary):**
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
* **Kept:** no code change for the leakage points 1-4 (the look-ahead risk was added to notebook 02, section 6);
  the ddof fix in notebook 02 and one sentence in notebook 04 after prompt 2.
