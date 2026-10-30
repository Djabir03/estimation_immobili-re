# House Price Prediction — notebook + application web

Deux livrables dans ce dossier :

1. le **notebook Colab** d'origine, exécuté localement ;
2. une **application web** pour saisir les données d'un logement et obtenir son prix estimé.

Notebook d'origine : `Project 4 : House Price Prediction.ipynb`
(https://colab.research.google.com/drive/1YFSkEzoJXTm3GEn3CZ6MUcg2kJktAHHc)

---

## 1. Lancer l'application web

Double-clic sur **`demarrer.bat`**, ou en ligne de commande :

```powershell
python app.py
```

Puis ouvrir **http://127.0.0.1:5000**

| Page | URL | Contenu |
|---|---|---|
| Nouvelle estimation | `/` | formulaire des 13 variables → prix estimé |
| Historique | `/history` | toutes les saisies enregistrées + export CSV |
| Modele | `/modele` | métriques, plages de valeurs, statistiques de la cible |

### Ce que fait la page de saisie

- **13 champs** regroupés en 3 sections (environnement, fiscalité, logement),
  pré-remplis avec les valeurs médianes du jeu de données ;
- **validation** : valeur obligatoire, numérique, et dans la plage observée
  en entraînement (message d'erreur sous le champ concerné) ;
- **résultat** : prix en dollars, fourchette indicative (± MAE), écart à la
  médiane du marché, et tableau récapitulatif des valeurs utilisées ;
- **enregistrement automatique** de chaque estimation dans
  `data/predictions.csv` (visible dans l'historique, exportable) ;
- **import par lot** : un CSV de plusieurs logements → une estimation par ligne
  (modèle téléchargeable depuis la page, exemple dans
  `data/logements_exemple.csv`).

L'application tourne **en local uniquement** : aucune donnée n'est envoyée
sur internet.

---

## 2. Structure des fichiers

| Fichier | Rôle |
|---|---|
| `app.py` | application Flask (formulaire, prédiction, historique, import CSV) |
| `train_model.py` | entraîne le XGBoost et écrit `model/` |
| `model/house_price_model.joblib` | modèle sérialisé utilisé par le web |
| `model/metadata.json` | features, bornes, médianes, métriques |
| `templates/` | pages HTML (`index`, `result`, `history`, `batch`, `model`) |
| `static/style.css` | feuille de style |
| `data/predictions.csv` | historique des estimations (créé à la volée) |
| `data/boston.csv` | cache du dataset Boston |
| `boston_compat.py` | remplace `sklearn.datasets.load_boston()` (retiré de sklearn ≥ 1.2) |
| `tests_routes.py` | tests automatiques des routes (`python tests_routes.py`) |
| `demarrer.bat` | lancement en un clic (Windows) |

---

## 3. Le notebook

| Fichier | Rôle |
|---|---|
| `house_price_prediction_original.ipynb` | copie brute du notebook Colab |
| `house_price_prediction.ipynb` | copie **exécutée** localement (sorties + figures) |
| `house_price_prediction.html` | rendu HTML lisible |
| `prepare_notebook.py` | régénère le notebook exécutable depuis l'original |

### Pourquoi un correctif ?

Le notebook utilise `sklearn.datasets.load_boston()`, **retiré de
scikit-learn ≥ 1.2** (problème éthique sur la variable `B`). La cellule a été
remplacée par :

```python
try:
    house_price_dataset = sklearn.datasets.load_boston()   # sklearn < 1.2
except (AttributeError, ImportError):
    from boston_compat import load_boston                  # fallback
    house_price_dataset = load_boston()
```

`boston_compat.load_boston()` reproduit l'API d'origine (`data`, `target`,
`feature_names`) depuis le dataset OpenML `boston` (id 531), mis en cache dans
`data/boston.csv`. Les valeurs sont identiques à celles du notebook d'origine
(`CRIM=0.00632`, `MEDV=24.0`, …). Le reste du code n'a **pas** été modifié.

Exécution :

```powershell
python prepare_notebook.py
python -m nbconvert --to notebook --execute --inplace house_price_prediction.ipynb
python -m nbconvert --to html house_price_prediction.ipynb
```

---

## 4. Installation

```powershell
python -m pip install -r requirements.txt
```

(`requirements.txt` : numpy, pandas, matplotlib, seaborn, scikit-learn,
xgboost, flask, nbconvert, ipykernel)

---

## 5. Résultats

| Jeu | R² | MAE |
|---|---|---|
| Entraînement (404 lignes) | 0.999998 | 0.0081 k$ |
| Test (102 lignes) | 0.877109 | 2.2488 k$ |

Le R² proche de 1 en entraînement est attendu : un `XGBRegressor` non réglé
mémorise fortement les données d'entraînement (surapprentissage). L'écart avec
le score de test (0.88) est la vraie mesure de performance — c'est cette valeur
qui est affichée dans l'interface.

Pour aller plus loin : réglage des hyperparamètres (`max_depth`,
`learning_rate`, `n_estimators`), validation croisée, et remplacement du jeu
Boston par un jeu plus actuel (California housing, Ames housing).
