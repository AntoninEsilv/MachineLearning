"""Evaluation protocol (owner: Enki).

Main metric: PR-AUC (average precision), because only 11 % of the calls are a yes.
Business metric: share of all subscribers reached when the bank calls the top 20 % of the list.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from common import SEED


def recall_at_top(y, score, frac=0.2):
    """Share of all the yes that are in the top `frac` of clients ranked by score.

    Ties: if several clients have the same score at the cut, we count the expected number of yes among
    the ones we pick (as if picked at random). Without this, a model with many equal scores (dummy, tree)
    would be judged on the row order, which is the time order here (see AI_AUDIT).
    """
    y, score = np.asarray(y), np.asarray(score)
    n_top = int(np.ceil(frac * len(y)))
    cut = np.sort(score)[::-1][n_top - 1]
    above = score > cut
    tied = score == cut
    n_left = n_top - above.sum()
    reached = y[above].sum() + n_left * y[tied].mean()
    return reached / y.sum()


def lift_at_top(y, score, frac=0.2):
    """How many times better than calling random clients (random gives recall = frac)."""
    return recall_at_top(y, score, frac) / frac


def ranking_metrics(y, score):
    return {"pr_auc": average_precision_score(y, score),
            "roc_auc": roc_auc_score(y, score),
            "recall_top20": recall_at_top(y, score, 0.2)}


def threshold_metrics(y, score, threshold):
    pred = (np.asarray(score) >= threshold).astype(int)
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "f1": f1_score(y, pred)}


def best_f1_threshold(y, score):
    """Threshold that maximises F1. Must be chosen on train (out-of-fold scores), never on test."""
    grid = np.unique(np.quantile(score, np.linspace(0.5, 0.995, 200)))
    f1s = [f1_score(y, (score >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def bootstrap_ci(y, score, metric, n_boot=1000, alpha=0.05, seed=SEED):
    """95 % confidence interval of a metric on the test set (resampling the test rows)."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(metric(y[idx], score[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(y, score_a, score_b, metric, n_boot=1000, seed=SEED):
    """Is model a really better than model b on the test set?

    Same resampled rows for both models; returns the mean difference, its 95 % CI and the
    share of resamples where a is not better (a one-sided bootstrap p-value).
    """
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(score_a), np.asarray(score_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    diffs = np.array(diffs)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(diffs.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_not_better": float((diffs <= 0).mean())}


def cv_summary(search_or_scores, name):
    """One line per model from cross_validate output: mean and std over the 5 folds."""
    s = search_or_scores
    row = {"model": name}
    for k in ["test_pr_auc", "test_roc_auc", "train_pr_auc"]:
        if k in s:
            row[k.replace("test_", "cv_") + "_mean"] = np.mean(s[k])
            row[k.replace("test_", "cv_") + "_std"] = np.std(s[k])
    if "fit_time" in s:
        row["fit_time_s"] = np.mean(s["fit_time"])
    return row


SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


def gain_curve(y, score):
    """x = share of clients called, y = share of subscribers reached."""
    y = np.asarray(y)
    # ties broken at random (not by row order, which is the time order)
    tie_break = np.random.default_rng(SEED).random(len(y))
    order = np.lexsort((tie_break, -np.asarray(score)))
    reached = np.cumsum(y[order]) / y.sum()
    called = np.arange(1, len(y) + 1) / len(y)
    return called, reached


def results_table(rows):
    return pd.DataFrame(rows).set_index("model").round(4)
