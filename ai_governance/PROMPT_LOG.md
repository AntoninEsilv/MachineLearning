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

### P2. *(to write by Enki)* constrained prompt
Write here a second, constrained prompt you really send, for example to audit notebook 02, and what it found.
