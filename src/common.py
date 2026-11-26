"""Shared helpers for the modelling notebooks.

The split and the preprocessing come from Vivien's module (code/step2_split.py,
code/step2_preprocessing.py): every model uses the same train.csv / test.csv and the same
preprocessor. This file only adds the CV folds and the temporal split.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "code"))
from step2_preprocessing import (CATEGORICAL, FEATURES, NUMERIC, SEED, TARGET,  # noqa: E402
                                 build_preprocessor)

DATASET = ROOT / "dataset"
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
# row_order is NOT a feature: it is the position in time, the share of yes goes from 3 % to 46 %
# along it, so a model would learn "late call = yes". We only keep it for the temporal split.
assert "row_order" not in FEATURES and "duration" not in FEATURES


def load_clean():
    return pd.read_csv(DATASET / "processed" / "bank_clean.csv")


def random_split(df=None):
    """Main split (made once by code/step2_split.py): stratified 80/20, seed 42.
    The test set is only used in notebook 04."""
    train = pd.read_csv(DATASET / "train.csv")
    test = pd.read_csv(DATASET / "test.csv")
    assert set(train.row_order).isdisjoint(test.row_order)
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


def make_preprocessor():
    """Encoding + scaling, always inside a Pipeline so it is fitted on the training folds only."""
    return build_preprocessor()


def set_seed():
    np.random.seed(SEED)
