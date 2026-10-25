"""
ablation.py : quelles caractéristiques généralisent d'un hôpital à l'autre ?

Pour chaque groupe de caractéristiques et pour 2 modèles, on mesure :
  - l'AUC en validation croisée sur toutes les images (même distribution),
  - l'AUC en entraînant sur un hôpital et en testant sur l'autre (avec un
    intervalle de confiance à 95 % par bootstrap).

Utilisation (après `python features.py`) :
    python ablation.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from classifier import BASE, META_COLS, SEED, make_models


def groups_of(cols):
    return {
        "forme": [c for c in cols if c in ("n_regions", "area_frac", "area_asym",
                                           "bbox_aspect", "solidity")],
        "intensité": [c for c in cols if c.startswith("int_")
                      or c in ("lung_vs_body", "frac_bright")],
        "zones": [c for c in cols if c.startswith("zone_") or c == "lr_mean_diff"],
        "texture": [c for c in cols if c.startswith(("glcm_", "lbp_"))],
    }


def auc_ci(y, p, n_boot=200, seed=0):
    """AUC et intervalle de confiance à 95 % (bootstrap)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return roc_auc_score(y, p), lo, hi


def main():
    ap = argparse.ArgumentParser(description="Ablation de caractéristiques")
    ap.add_argument("--features", default=str(BASE / "results" / "features.csv"))
    ap.add_argument("--out", default=str(BASE / "results"))
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    y = df["ptb"].astype(int)
    cols = [c for c in df.columns if c not in META_COLS]
    g = groups_of(cols)
    sets = {
        "toutes": g["forme"] + g["intensité"] + g["zones"] + g["texture"],
        "sans texture": g["forme"] + g["intensité"] + g["zones"],
        "texture seule": g["texture"],
        "sans forme": g["intensité"] + g["zones"] + g["texture"],
        "intensité + zones": g["intensité"] + g["zones"],
        "zones seules": g["zones"],
    }
    counties = list(df["county"].dropna().unique())
    if len(counties) != 2:
        raise SystemExit("Il faut exactement 2 valeurs de 'county' pour ce contrôle.")
    a, b = counties
    print(f"Hôpitaux : {a} ({int((df.county == a).sum())}) et {b} ({int((df.county == b).sum())})")

    models = make_models()
    models = {k: models[k] for k in ("Régression logistique", "Random Forest")}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for set_name, feats in sets.items():
        feats = [c for c in feats if c in df.columns]
        X = df[feats]
        for m_name, model in models.items():
            p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            row = {"caractéristiques": set_name, "n": len(feats), "modèle": m_name,
                   "AUC validation croisée": round(roc_auc_score(y, p), 3)}
            for tr_name, te_name in ((a, b), (b, a)):
                tr, te = df.county == tr_name, df.county == te_name
                mdl = clone(model).fit(X[tr], y[tr])
                auc, lo, hi = auc_ci(y[te], mdl.predict_proba(X[te])[:, 1])
                row[f"AUC {tr_name[:4]}->{te_name[:4]}"] = f"{auc:.2f} [{lo:.2f}-{hi:.2f}]"
            rows.append(row)
            print(f"  {set_name:18s} {m_name:22s} fait")
    res = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "ablation.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + res.to_string(index=False))
    print(f"\nFichier : {out / 'ablation.csv'}")
    print("\nLecture : un AUC inter-hôpitaux proche de 0.50 = le modèle ne généralise pas ;")
    print("l'intervalle entre crochets est l'incertitude (95 %).")


if __name__ == "__main__":
    main()