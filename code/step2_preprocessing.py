"""
Préparation des features - Bank Telemarketing (Step 2)

Ce module définit le préprocesseur (encodage + normalisation).
Il N'EST PAS appliqué au dataset avant le split : il est intégré dans un
Pipeline scikit-learn, donc ajusté (fit) uniquement sur les données
d'entraînement de chaque fold de validation croisée -> pas de fuite de données.

Usage :
    from step2_preprocessing import load_split, build_preprocessor
    X_train, y_train, X_test, y_test = load_split()
    model = Pipeline([("prep", build_preprocessor()), ("clf", LogisticRegression(...))])
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"

SEED = 42
TARGET = "y"

# Colonne technique (index de ligne d'origine) : aucune valeur prédictive -> exclue
ID_COLS = ["row_order"]

CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
NUMERIC = [
    "age", "previous", "pdays_clean", "campaign_log",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
BINARY = ["prev_contacted"]  # déjà 0/1

FEATURES = CATEGORICAL + NUMERIC + BINARY


def build_preprocessor() -> ColumnTransformer:
    """One-hot sur les catégorielles ("unknown" gardé comme modalité),
    standardisation des numériques, binaire laissé tel quel."""
    return ColumnTransformer(
        transformers=[
            # modalités rares (< 20 lignes, ex. default=yes : 3 lignes) regroupées, comme annoncé dans le notebook 01
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                  sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
            ("bin", "passthrough", BINARY),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def load_split():
    """Charge train/test déjà séparés et renvoie X_train, y_train, X_test, y_test."""
    train = pd.read_csv(DATASET_DIR / "train.csv")
    test = pd.read_csv(DATASET_DIR / "test.csv")
    return train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
