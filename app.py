"""Application web locale : estimation du prix d'un logement (Boston Housing).

Lancement :
    python app.py
puis ouvrir http://127.0.0.1:5000
"""

import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "house_price_model.joblib"
META_PATH = ROOT / "model" / "metadata.json"
HISTORY_PATH = ROOT / "data" / "predictions.csv"

# Regroupement des champs pour l'affichage du formulaire
FEATURE_GROUPS = [
    (
        "Environnement du quartier",
        "Qualite de vie, pollution et desserte de la zone.",
        ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "DIS"],
    ),
    (
        "Fiscalite et accessibilite",
        "Charges locales et proximite des axes routiers.",
        ["RAD", "TAX", "PTRATIO"],
    ),
    (
        "Le logement",
        "Caracteristiques physiques du bien et du voisinage.",
        ["RM", "AGE", "LSTAT", "B"],
    ),
]

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "house-price-prediction-local")


# --------------------------------------------------------------------------
# Chargement du modele
# --------------------------------------------------------------------------
def load_artifacts():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise SystemExit(
            "Modele introuvable. Lancez d'abord :  python train_model.py"
        )
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


MODEL, META = load_artifacts()
FEATURES = META["features"]
FEATURE_NAMES = [f["name"] for f in FEATURES]
FEATURE_BY_NAME = {f["name"]: f for f in FEATURES}
METRICS = META["metrics"]


@app.context_processor
def inject_globals():
    """`meta` et `metrics` disponibles dans tous les templates."""
    return {"meta": META, "metrics": METRICS}


def format_price(value_thousands: float) -> str:
    """24.7 -> '24 700 $'"""
    return f"{round(value_thousands * 1000):,} $".replace(",", " ")


app.jinja_env.filters["prix"] = format_price


def predict(features: dict) -> float:
    row = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)
    return float(MODEL.predict(row)[0])


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def parse_form(form) -> tuple[dict, dict]:
    """Retourne (valeurs, erreurs)."""
    values, errors = {}, {}

    for feat in FEATURES:
        name = feat["name"]
        raw = (form.get(name) or "").strip().replace(",", ".")

        if raw == "":
            errors[name] = "Champ obligatoire."
            values[name] = feat["median"]
            continue

        try:
            value = float(raw)
        except ValueError:
            errors[name] = "Valeur numerique attendue."
            values[name] = feat["median"]
            continue

        if value < feat["min"] or value > feat["max"]:
            errors[name] = (
                f"Hors plage observee ({feat['min']:g} a {feat['max']:g})."
            )
        values[name] = value

    return values, errors


# --------------------------------------------------------------------------
# Historique
# --------------------------------------------------------------------------
HISTORY_FIELDS = ["horodatage"] + FEATURE_NAMES + ["prix_milliers", "prix_dollars"]


def append_history(values: dict, price: float) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not HISTORY_PATH.exists()

    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **{name: values[name] for name in FEATURE_NAMES},
                "prix_milliers": round(price, 3),
                "prix_dollars": round(price * 1000),
            }
        )


def read_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return list(reversed(rows))


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
def render_form(values: dict, errors: dict, status: int = 200):
    return (
        render_template(
            "index.html",
            groups=FEATURE_GROUPS,
            features=FEATURE_BY_NAME,
            feature_names=FEATURE_NAMES,
            values=values,
            errors=errors,
            meta=META,
            metrics=METRICS,
        ),
        status,
    )


@app.route("/")
def index():
    defaults = {f["name"]: f["median"] for f in FEATURES}
    return render_form(defaults, {})


@app.route("/predict", methods=["POST"])
def predict_route():
    values, errors = parse_form(request.form)

    if errors:
        return render_form(values, errors, status=400)

    price = predict(values)
    append_history(values, price)

    return render_template(
        "result.html",
        price=price,
        price_text=format_price(price),
        low_text=format_price(max(price - METRICS["mae_test"], 0)),
        high_text=format_price(price + METRICS["mae_test"]),
        values=values,
        features=FEATURE_BY_NAME,
        meta=META,
        metrics=METRICS,
        target_median_text=format_price(META["target"]["median"]),
    )


@app.route("/history")
def history():
    rows = read_history()
    return render_template(
        "history.html",
        rows=rows,
        features=FEATURES,
        meta=META,
        total=len(rows),
    )


@app.route("/history.csv")
def history_csv():
    if not HISTORY_PATH.exists():
        return "Aucun historique disponible.", 404
    return send_file(
        HISTORY_PATH,
        mimetype="text/csv",
        as_attachment=True,
        download_name="predictions.csv",
    )


@app.route("/history/clear", methods=["POST"])
def history_clear():
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    flash("Historique efface.")
    return redirect(url_for("history"))


@app.route("/batch/template")
def batch_template():
    """Modele de CSV : en-tete + une ligne mediane."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(FEATURE_NAMES)
    writer.writerow([FEATURE_BY_NAME[n]["median"] for n in FEATURE_NAMES])
    writer.writerow([FEATURE_BY_NAME[n]["median"] for n in FEATURE_NAMES])

    data = io.BytesIO(buffer.getvalue().encode("utf-8"))
    return send_file(
        data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="modele_logements.csv",
    )


@app.route("/batch", methods=["POST"])
def batch():
    """Prediction par lot a partir d'un CSV."""
    upload = request.files.get("fichier")
    if not upload or not upload.filename:
        flash("Aucun fichier selectionne.")
        return redirect(url_for("index"))

    try:
        text = upload.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        flash("Le fichier doit etre encode en UTF-8.")
        return redirect(url_for("index"))

    reader = csv.DictReader(io.StringIO(text))
    header = [(c or "").strip() for c in (reader.fieldnames or [])]
    mapping = {c.lower(): c for c in header}

    missing = [n for n in FEATURE_NAMES if n.lower() not in mapping]
    if missing:
        flash(
            "Colonnes manquantes dans le CSV : "
            + ", ".join(missing)
            + ". Colonnes attendues : "
            + ", ".join(FEATURE_NAMES)
        )
        return redirect(url_for("index"))

    results, ignored = [], 0
    for line_no, row in enumerate(reader, start=2):
        try:
            values = {
                name: float(str(row[mapping[name.lower()]]).strip().replace(",", "."))
                for name in FEATURE_NAMES
            }
        except (TypeError, ValueError):
            ignored += 1
            continue
        price = predict(values)
        results.append({"line": line_no, "values": values, "price": price})

    if not results:
        flash("Aucune ligne exploitable dans le fichier.")
        return redirect(url_for("index"))

    return render_template(
        "batch.html",
        results=results,
        ignored=ignored,
        features=FEATURE_BY_NAME,
        feature_names=FEATURE_NAMES,
        meta=META,
        metrics=METRICS,
    )


@app.route("/modele")
def model_info():
    return render_template(
        "model.html",
        meta=META,
        metrics=METRICS,
        features=FEATURES,
        n_history=len(read_history()),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Application disponible sur http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
