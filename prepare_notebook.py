"""Prepare une copie executable du notebook Colab.

- reprend le notebook d'origine tel quel ;
- remplace uniquement la cellule `load_boston()` par un appel tolerant a
  la suppression de cette fonction dans scikit-learn >= 1.2 ;
- vide les sorties et les compteurs d'execution avant de relancer.
"""

import json
from pathlib import Path

SRC = Path(__file__).with_name("house_price_prediction_original.ipynb")
DST = Path(__file__).with_name("house_price_prediction.ipynb")

PATCHED_CELL = """\
# `load_boston` a ete retire de scikit-learn >= 1.2 : fallback compatible
try:
    house_price_dataset = sklearn.datasets.load_boston()
except (AttributeError, ImportError):
    from boston_compat import load_boston
    house_price_dataset = load_boston()
"""


def to_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    patched = False

    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "load_boston()" in src and "sklearn.datasets.load_boston" in src:
            cell["source"] = to_source(PATCHED_CELL)
            patched = True

        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            cell.get("metadata", {}).pop("executionInfo", None)

    if not patched:
        raise SystemExit("Cellule load_boston introuvable")

    nb.setdefault("metadata", {})["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb["metadata"]["language_info"] = {"name": "python"}

    DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook prepare : {DST.name} ({len(nb['cells'])} cellules)")


if __name__ == "__main__":
    main()
