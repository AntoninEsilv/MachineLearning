"""Registry of the final models compared in notebook 04.

Each entry is a function that returns an unfitted sklearn Pipeline with its tuned
hyperparameters. The tuned values are read from results/best_params_<name>.json, written by
the notebook that tuned the model (02 for the baseline and AutoML v0.1, 03 for step 3).

To add a step 3 model: tune it in notebook 03 or 06, save its best params with save_best_params(),
then add one line to BUILDERS below. Notebook 04 evaluates every model of BUILDERS that has a
results/best_params_<name>.json file (so a model that was not selected in CV has no file).
"""
import json

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from common import RESULTS, SEED, cv_folds, make_preprocessor


def save_best_params(name, params):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"best_params_{name}.json", "w") as f:
        json.dump(params, f, indent=2)


def load_best_params(name):
    path = RESULTS / f"best_params_{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# unfitted pipelines with default settings, the tuning notebooks set the params with set_params
def pipe_dummy():
    return Pipeline([("prep", make_preprocessor()), ("model", DummyClassifier(strategy="prior"))])


def pipe_logreg():
    return Pipeline([("prep", make_preprocessor()),
                     ("model", LogisticRegression(max_iter=5000, random_state=SEED))])


def pipe_tree():
    return Pipeline([("prep", make_preprocessor()),
                     ("model", DecisionTreeClassifier(random_state=SEED))])


def pipe_knn():
    return Pipeline([("prep", make_preprocessor()), ("model", KNeighborsClassifier())])


def pipe_nb():
    return Pipeline([("prep", make_preprocessor()), ("model", GaussianNB())])


def pipe_linsvc():
    # no predict_proba, decision_function is enough to rank clients (PR-AUC only needs an order)
    return Pipeline([("prep", make_preprocessor()),
                     ("model", LinearSVC(random_state=SEED, max_iter=20000))])


# step 3 models (tuned in notebook 03)
def pipe_rf():
    return Pipeline([("prep", make_preprocessor()),
                     ("model", RandomForestClassifier(random_state=SEED, n_jobs=-1))])


def pipe_lgbm():
    # n_jobs=1: the cross-validation already runs the folds in parallel (n_jobs=-1). If LightGBM also
    # uses all the cores, the threads fight each other and it is about 60 times slower
    # (measured: 102 s instead of 1.7 s for one 5-fold CV).
    return Pipeline([("prep", make_preprocessor()),
                     ("model", LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1))])


# step 3, part 2 (notebook 06)
# the ensembles reuse the tuned base models, each with its own preprocessor inside its pipeline
ENSEMBLE_BASE = ["logreg_baseline", "random_forest", "lightgbm"]


def pipe_stacking():
    # stacked generalization (Wolpert 1992): the meta-model is trained on out-of-fold scores of the
    # base models (cv=...), never on scores of rows they were trained on, otherwise it would trust the
    # most overfitted model
    return StackingClassifier(estimators=[(n, get_model(n)) for n in ENSEMBLE_BASE],
                              final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
                              cv=cv_folds(), stack_method="predict_proba")


def pipe_voting():
    # soft voting: mean of the probabilities of the base models, no extra training
    return VotingClassifier(estimators=[(n, get_model(n)) for n in ENSEMBLE_BASE], voting="soft")


def pipe_mlp():
    # model outside the course: multilayer perceptron (one-hot + standardised inputs, ReLU, sigmoid output)
    # early_stopping stays False: in scikit-learn it monitors the ACCURACY of a validation split, which is
    # useless with 11 % of yes; the number of epochs (max_iter) is tuned instead
    return Pipeline([("prep", make_preprocessor()),
                     ("model", MLPClassifier(random_state=SEED, early_stopping=False))])


BUILDERS = {
    "dummy": pipe_dummy,
    "logreg_baseline": pipe_logreg,
    "decision_tree": pipe_tree,
    "knn": pipe_knn,
    "naive_bayes": pipe_nb,
    "linear_svc": pipe_linsvc,
    "random_forest": pipe_rf,
    "lightgbm": pipe_lgbm,
    "stacking": pipe_stacking,
    "soft_voting": pipe_voting,
    "mlp": pipe_mlp,
}


def get_model(name):
    pipe = BUILDERS[name]()
    params = load_best_params(name)
    if params:
        pipe.set_params(**params)
    return pipe


def score(pipe, X):
    """Score used to rank clients: probability of yes, or decision function if no probability."""
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return pipe.decision_function(X)
