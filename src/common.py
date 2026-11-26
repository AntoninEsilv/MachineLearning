"""Shared helpers: load the clean data, split it, build the preprocessing.

Every notebook imports from here so that all the models use the same features,
the same test set and the same preprocessing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "processed" / "bank_clean.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
NUMERIC = ["age", "campaign_log", "pdays_clean", "prev_contacted", "previous",
           "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "y"
# row_order is NOT a feature: it is the position in time, the share of yes goes from 3 % to 46 %
# along it, so a model would learn "late call = yes". We only keep it for the temporal split.


def load_clean():
    df = pd.read_csv(DATA)
    assert "duration" not in df.columns  # leak, removed in notebook 01
    return df


def random_split(df, test_size=0.2):
    """Main split: stratified 80/20, the test set is only used in notebook 04."""
    train, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=SEED)
    return train, test


def temporal_split(df, test_size=0.2):
    """Robustness check: train on the oldest 80 % of calls, test on the most recent 20 %."""
    df = df.sort_values("row_order")
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut], df.iloc[cut:]


def xy(df):
    return df[FEATURES], df[TARGET].to_numpy()


def cv_folds():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def make_preprocessor(scale=True):
    """One-hot for categories (rare ones < 20 rows grouped together), scaling for numbers.

    It is put inside a Pipeline, so it is fitted on the training folds only (no leakage).
    """
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("cat", ohe, CATEGORICAL), ("num", num, NUMERIC)])


def set_seed():
    np.random.seed(SEED)
