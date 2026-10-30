"""Entraine le modele XGBoost et le sauvegarde pour l'application web.

Produit :
  model/house_price_model.joblib  -> modele entraine sur les 506 lignes
  model/metadata.json             -> features, bornes, medianes, metriques
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from boston_compat import FEATURE_NAMES, TARGET_NAME, load_boston

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
META_PATH = MODEL_DIR / "metadata.json"

# Libelles + aides affiches dans le formulaire (FR)
FEATURE_INFO = {
    "CRIM": ("Taux de criminalite par ville", "taux", 2),
    "ZN": ("% de terrains residentiels > 25 000 pieds2", "%", 2),
    "INDUS": ("% de surfaces commerciales non-retail", "%", 2),
    "CHAS": ("Borde la riviere Charles (0 = non, 1 = oui)", "", 0),
    "NOX": ("Concentration en oxydes d'azote", "ppm", 3),
    "RM": ("Nombre moyen de pieces par logement", "pieces", 2),
    "AGE": ("% de logements construits avant 1940", "%", 2),
    "DIS": ("Distance ponderee aux 5 centres d'emploi", "miles", 3),
    "RAD": ("Indice d'accessibilite aux autoroutes", "indice", 1),
    "TAX": ("Taux de taxe fonciere pour 10 000 $", "$/10k$", 1),
    "PTRATIO": ("Ratio eleves / enseignants par ville", "ratio", 2),
    "B": ("1000(Bk - 0.63)^2 ou Bk = % de residents noirs", "indice", 2),
    "LSTAT": ("% de population a bas statut social", "%", 2),
}


def main() -> None:
    bunch = load_boston()
    X = np.asarray(bunch.data, dtype=float)
    y = np.asarray(bunch.target, dtype=float)

    # Metriques honnetes : split identique au notebook
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=2)
    eval_model = XGBRegressor()
    eval_model.fit(X_tr, y_tr)
    pred = eval_model.predict(X_te)
    metrics = {
        "r2_test": round(float(r2_score(y_te, pred)), 6),
        "mae_test": round(float(mean_absolute_error(y_te, pred)), 4),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
    }

    # Modele final : toutes les donnees
    model = XGBRegressor()
    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    features = []
    for idx, name in enumerate(FEATURE_NAMES):
        label, unit, step = FEATURE_INFO[name]
        col = X[:, idx]
        features.append(
            {
                "name": name,
                "label": label,
                "unit": unit,
                "step": step,
                "min": float(col.min()),
                "max": float(col.max()),
                "median": round(float(np.median(col)), 3),
                "mean": round(float(col.mean()), 3),
                "is_binary": bool(name == "CHAS"),
            }
        )

    meta = {
        "target_name": TARGET_NAME,
        "target_unit": "milliers de dollars (1000 $)",
        "features": features,
        "metrics": metrics,
        "n_samples": int(X.shape[0]),
        "target": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": round(float(y.mean()), 3),
            "median": round(float(np.median(y)), 3),
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Modele sauvegarde      : {MODEL_PATH}")
    print(f"Metadonnees sauvegardees: {META_PATH}")
    print(f"R2 test = {metrics['r2_test']} | MAE test = {metrics['mae_test']} k$")


if __name__ == "__main__":
    main()
