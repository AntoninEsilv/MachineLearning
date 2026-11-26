"""
Test feature selection without using the test set for model or subset selection.

The preprocessing and each selector are inside the cross-validation pipeline.
Candidate subsets are selected only from the 5 training-fold CV scores. The test
set is evaluated once, after the best CV method has been chosen.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from common import RESULTS, FIGURES, cv_folds, make_preprocessor, random_split, set_seed, xy  # noqa: E402
from evaluation import SCORING, bootstrap_ci, paired_bootstrap_diff, ranking_metrics  # noqa: E402


set_seed()
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def mi_score(X, y):
    """Deterministic mutual information score for the sklearn pipeline."""
    return mutual_info_classif(X, y, random_state=42)


def make_logreg_selector(name):
    if name == "all_features":
        return None
    if name == "mi_50":
        return SelectPercentile(score_func=mi_score, percentile=50)
    if name == "mi_75":
        return SelectPercentile(score_func=mi_score, percentile=75)
    if name == "l1_median":
        return SelectFromModel(
            LogisticRegression(
                l1_ratio=1.0,
                C=0.1,
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            ),
            threshold="median",
        )
    if name == "l1_strict":
        return SelectFromModel(
            LogisticRegression(
                l1_ratio=1.0,
                C=0.1,
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            ),
            threshold="1.25*mean",
        )
    raise ValueError(f"Unknown feature-selection method: {name}")


def make_pipeline(selector=None):
    steps = [("prep", make_preprocessor())]
    if selector is not None:
        steps.append(("select", selector))
    steps.append(
        (
            "model",
            LogisticRegression(
                C=0.1,
                class_weight=None,
                max_iter=5000,
                random_state=42,
            ),
        )
    )
    return Pipeline(steps)


def model_score(pipeline, X):
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def main():
    train, test = random_split()
    X_train, y_train = xy(train)
    X_test, y_test = xy(test)

    cv_rows = []
    methods = ["all_features", "mi_50", "mi_75", "l1_median", "l1_strict"]
    pipelines = {}

    for method in methods:
        pipeline = make_pipeline(make_logreg_selector(method))
        pipelines[method] = pipeline
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_folds(),
            scoring=SCORING,
            return_train_score=True,
            n_jobs=-1,
        )
        cv_rows.append(
            {
                "method": method,
                "cv_pr_auc_mean": scores["test_pr_auc"].mean(),
                "cv_pr_auc_std": scores["test_pr_auc"].std(),
                "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
                "cv_roc_auc_std": scores["test_roc_auc"].std(),
                "train_pr_auc_mean": scores["train_pr_auc"].mean(),
                "fit_time_s": scores["fit_time"].mean(),
            }
        )

    cv_results = pd.DataFrame(cv_rows).sort_values(
        "cv_pr_auc_mean", ascending=False
    )
    cv_results.to_csv(RESULTS / "feature_selection_cv.csv", index=False)

    best_method = cv_results.iloc[0]["method"]
    print("Cross-validation results:")
    print(cv_results.round(4).to_string(index=False))
    print("Best CV method:", best_method)

    test_rows = []
    test_scores = {}
    baseline = make_pipeline()
    baseline.fit(X_train, y_train)
    baseline_score = model_score(baseline, X_test)
    test_scores["all_features"] = baseline_score

    baseline_metrics = ranking_metrics(y_test, baseline_score)
    test_rows.append(
        {
            "method": "all_features",
            **baseline_metrics,
            "pr_auc_ci_low": bootstrap_ci(y_test, baseline_score, average_precision_score)[0],
            "pr_auc_ci_high": bootstrap_ci(y_test, baseline_score, average_precision_score)[1],
        }
    )

    selected_names = {}
    for method in methods[1:]:
        pipeline = pipelines[method]
        pipeline.fit(X_train, y_train)
        test_score = model_score(pipeline, X_test)
        test_scores[method] = test_score

        metrics = ranking_metrics(y_test, test_score)
        ci_low, ci_high = bootstrap_ci(
            y_test, test_score, average_precision_score
        )
        test_rows.append(
            {
                "method": method,
                **metrics,
                "pr_auc_ci_low": ci_low,
                "pr_auc_ci_high": ci_high,
            }
        )

        if method == best_method:
            prep = pipeline.named_steps["prep"]
            transformed_names = prep.get_feature_names_out()
            support = pipeline.named_steps["select"].get_support()
            selected_names[method] = list(transformed_names[support])

    test_results = pd.DataFrame(test_rows).sort_values("pr_auc", ascending=False)
    test_results.to_csv(RESULTS / "feature_selection_test.csv", index=False)

    selected_pipeline = pipelines[best_method]
    selected_pipeline.fit(X_train, y_train)
    if best_method != "all_features":
        transformed_names = selected_pipeline.named_steps["prep"].get_feature_names_out()
        support = selected_pipeline.named_steps["select"].get_support()
        selected_names[best_method] = list(transformed_names[support])

        pd.DataFrame(
            {"selected_feature": selected_names[best_method]}
        ).to_csv(RESULTS / "feature_selection_names.csv", index=False)
        print(
            f"Selected features ({len(selected_names[best_method])}):",
            ", ".join(selected_names[best_method]),
        )
    else:
        pd.DataFrame(
            {"selected_feature": X_train.columns.tolist()}
        ).to_csv(RESULTS / "feature_selection_names.csv", index=False)

    paired = paired_bootstrap_diff(
        y_test,
        test_scores[best_method],
        baseline_score,
        average_precision_score,
    )
    pd.DataFrame([{"best_method": best_method, **paired}]).to_csv(
        RESULTS / "feature_selection_paired.csv", index=False
    )

    print("Test results:")
    print(test_results.round(4).to_string(index=False))
    print("Paired bootstrap vs all_features:", paired)

    plot = cv_results.sort_values("cv_pr_auc_mean")
    plt.figure(figsize=(9, 5))
    plt.barh(
        plot["method"],
        plot["cv_pr_auc_mean"],
        xerr=plot["cv_pr_auc_std"],
        color="steelblue",
    )
    plt.xlabel("Cross-validation PR-AUC")
    plt.title("Feature-selection methods (5-fold CV)")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_selection_cv.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
