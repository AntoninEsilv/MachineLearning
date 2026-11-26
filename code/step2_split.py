"""
Split du dataset nettoyé en train (80 %) / test (20 %) - Bank Telemarketing (Step 2)

- Split stratifié sur y : la proportion de "yes" (~11 %) est la même dans les deux jeux.
- Graine fixée (SEED = 42) : split reproductible.
- Le jeu de test est écrit une fois pour toutes et ne sert qu'à l'évaluation finale.
- Les données sont sauvegardées NON encodées (propres mais lisibles) ; l'encodage est
  fait dans un Pipeline (voir step2_preprocessing.py) pour éviter toute fuite de données.

Lancer depuis la racine du projet :  python code/step2_split.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from step2_preprocessing import DATASET_DIR, FEATURES, ID_COLS, SEED, TARGET

TEST_SIZE = 0.20


def main():
    df = pd.read_csv(DATASET_DIR / "processed/bank_clean.csv")

    # Contrôles de cohérence
    assert df[TARGET].isin([0, 1]).all(), "La cible doit être binaire 0/1"
    assert df.isna().sum().sum() == 0, "Il reste des valeurs manquantes"
    # Lignes aux features identiques (hors row_order) : ce sont des appels distincts
    # (la colonne duration a été retirée), on les conserve. Information seulement.
    n_same = df.drop(columns=ID_COLS).duplicated().sum()
    print(f"Info : {n_same} lignes ont des features identiques à une autre (appels distincts, conservées)\n")
    assert df["row_order"].is_unique, "row_order doit identifier une ligne de façon unique"
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    assert not missing, f"Colonnes manquantes : {missing}"

    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED, shuffle=True
    )
    train = train.sort_values("row_order").reset_index(drop=True)
    test = test.sort_values("row_order").reset_index(drop=True)

    # Aucune ligne commune entre train et test
    assert set(train["row_order"]).isdisjoint(test["row_order"])
    assert len(train) + len(test) == len(df)

    train.to_csv(DATASET_DIR / "train.csv", index=False)
    test.to_csv(DATASET_DIR / "test.csv", index=False)

    print(f"{'Jeu':<8}{'Lignes':>8}{'Part':>8}{'Taux de yes':>14}")
    for name, d in [("complet", df), ("train", train), ("test", test)]:
        print(f"{name:<8}{len(d):>8}{len(d)/len(df):>8.1%}{d[TARGET].mean():>14.2%}")
    print(f"\nFichiers écrits dans {DATASET_DIR}")


if __name__ == "__main__":
    main()
