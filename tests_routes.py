"""Tests rapides des routes de l'application (client Flask de test).

N'ecrit pas dans l'historique reel : les ecritures sont redirigees vers un
fichier temporaire.

    python tests_routes.py
"""

import io
import json
import re
import tempfile
from pathlib import Path

import app

# Historique isole
app.HISTORY_PATH = Path(tempfile.gettempdir()) / "predictions_test.csv"
app.HISTORY_PATH.unlink(missing_ok=True)

client = app.app.test_client()
meta = json.loads(Path("model/metadata.json").read_text(encoding="utf-8"))
med = {f["name"]: f["median"] for f in meta["features"]}

ok = True


def check(label, condition, detail=""):
    global ok
    ok = ok and condition
    print(f"{'OK  ' if condition else 'FAIL'} {label} {detail}")


# 1. Formulaire
r = client.get("/")
check("GET  /", r.status_code == 200)
check("13 champs presents", r.data.count(b'class="field ') == 13)

# 2. Prediction valide
r = client.post("/predict", data=med)
amount = re.search(rb'class="amount">([^<]+)<', r.data)
check("POST /predict", r.status_code == 200, "| prix = " + (amount.group(1).decode() if amount else "?"))

# 3. Valeurs invalides
bad = dict(med, CRIM="999", RM="abc")
r = client.post("/predict", data=bad)
check("POST /predict invalide", r.status_code == 400, "| erreurs = " + str(r.data.count(b'class="err"')))

# 4. Champs vides
r = client.post("/predict", data={})
check("POST /predict vide", r.status_code == 400, "| erreurs = " + str(r.data.count(b'class="err"')))

# 5. Import CSV (2 lignes valides + 1 cassee)
header = ",".join(med)
ligne = ",".join(str(v) for v in med.values())
csv_data = f"{header}\n{ligne}\n{ligne}\noops,bad\n"
r = client.post(
    "/batch",
    data={"fichier": (io.BytesIO(csv_data.encode()), "test.csv")},
    content_type="multipart/form-data",
)
check("POST /batch", r.status_code == 200, "| lignes = " + str(r.data.count(b"<tr>") - 1))

# 6. CSV incomplet -> message d'erreur
r = client.post(
    "/batch",
    data={"fichier": (io.BytesIO(b"CRIM,RM\n0.1,6.5\n"), "mauvais.csv")},
    content_type="multipart/form-data",
    follow_redirects=True,
)
check("POST /batch incomplet", b"Colonnes manquantes" in r.data)

# 7. Historique
r = client.get("/history")
check("GET  /history", r.status_code == 200)
r = client.get("/history.csv")
check("GET  /history.csv", r.status_code == 200, f"| {len(r.data)} octets")

# 8. Autres pages
check("GET  /modele", client.get("/modele").status_code == 200)
check("GET  /batch/template", client.get("/batch/template").status_code == 200)

print("\nRESULTAT :", "tous les tests passent" if ok else "echecs detectes")
raise SystemExit(0 if ok else 1)
