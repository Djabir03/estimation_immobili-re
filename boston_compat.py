"""Compatibilite pour `sklearn.datasets.load_boston()`.

`load_boston` a ete retire de scikit-learn depuis la version 1.2 (donnees
jugees ethiquement problematiques). Ce module reproduit l'ancienne API
(noms de colonnes, ordre des colonnes, target MEDV) en recuperant le jeu de
donnees Boston depuis OpenML, avec un cache local pour rester reproductible
hors ligne.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

FEATURE_NAMES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]
TARGET_NAME = "MEDV"

CACHE_FILE = Path(__file__).with_name("data") / "boston.csv"


def _fetch_boston_frame() -> pd.DataFrame:
    """Recupere le dataset Boston (506 lignes) via OpenML."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
    frame = bunch.frame.copy()

    # OpenML renvoie des noms de colonnes parfois prefixes (ex: "CHAS_1").
    rename = {}
    for col in frame.columns:
        if col == TARGET_NAME:
            continue
        for feat in FEATURE_NAMES:
            if col == feat or col.startswith(feat + "_"):
                rename[col] = feat
                break
    frame = frame.rename(columns=rename)

    missing = [c for c in FEATURE_NAMES + [TARGET_NAME] if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Colonnes manquantes dans le dataset OpenML: {missing}")

    frame = frame[FEATURE_NAMES + [TARGET_NAME]].apply(pd.to_numeric)
    return frame


def load_boston(return_X_y: bool = False):
    """Remplace `sklearn.datasets.load_boston()` (API identique)."""
    if CACHE_FILE.exists():
        frame = pd.read_csv(CACHE_FILE)
    else:
        frame = _fetch_boston_frame()
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(CACHE_FILE, index=False)

    data = frame[FEATURE_NAMES].to_numpy(dtype=float)
    target = frame[TARGET_NAME].to_numpy(dtype=float)

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        feature_names=list(FEATURE_NAMES),
        DESCR="Boston housing dataset (OpenML id 531), compatible load_boston.",
        filename=str(CACHE_FILE),
    )
