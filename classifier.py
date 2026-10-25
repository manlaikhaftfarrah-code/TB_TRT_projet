"""
classifier.py : classification normal / suspect TB à partir des caractéristiques.

Méthode :
  1. séparation 80 % entraînement / 20 % test (le test n'est utilisé qu'UNE fois),
  2. comparaison de 3 modèles par validation croisée (5 plis) sur l'entraînement,
  3. évaluation du meilleur modèle sur le test (sensibilité, spécificité, AUC...),
  4. contrôle : entraînement sur un "county", test sur l'autre,
  5. sauvegarde du modèle dans models/.

Choix par défaut (justifiés par ablation.py) : régression logistique, sans les
caractéristiques de forme (elles dépendent du cadrage de l'image de chaque hôpital).

Utilisation (après `python features.py`) :
    python classifier.py                    # choix par défaut
    python classifier.py --model auto       # meilleur modèle par AUC en validation croisée
    python classifier.py --with-shape       # garde aussi les caractéristiques de forme
"""
import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent
META_COLS = ["id", "ptb", "county", "erosion", "dice_seg"]
SEED = 0
SHAPE_COLS = ["n_regions", "area_frac", "area_asym", "bbox_aspect", "solidity"]


def make_models():
    prep = lambda: (SimpleImputer(strategy="median"), StandardScaler())
    return {
        "Régression logistique": make_pipeline(
            *prep(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "SVM (RBF)": make_pipeline(
            *prep(), CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=SEED),
                method="sigmoid", cv=3, ensemble=False)),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   random_state=SEED, n_jobs=1)),
    }


def metrics(y, proba, thr=0.5):
    """Métriques de classification (TB = classe positive)."""
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else np.nan
    auc = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else np.nan
    return {"accuracy": (tp + tn) / len(y), "sensibilité": sens, "spécificité": spec,
            "précision": prec, "F1": f1, "AUC": auc, "TP": tp, "FN": fn, "TN": tn, "FP": fp}


def main():
    ap = argparse.ArgumentParser(description="Classification normal / TB")
    ap.add_argument("--features", default=str(BASE / "results" / "features.csv"))
    ap.add_argument("--out", default=str(BASE / "results"))
    ap.add_argument("--models", default=str(BASE / "models"))
    ap.add_argument("--model", default="Régression logistique",
                    help='nom du modèle, ou "auto" pour prendre le meilleur en validation croisée')
    ap.add_argument("--with-shape", action="store_true",
                    help="garde les caractéristiques de forme (exclues par défaut)")
    args = ap.parse_args()

    out, mdir = Path(args.out), Path(args.models)
    out.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    if not args.with_shape:
        feat_cols = [c for c in feat_cols if c not in SHAPE_COLS]
    X, y = df[feat_cols], df["ptb"].astype(int)
    print(f"Images : {len(df)} | caractéristiques : {len(feat_cols)} | "
          f"TB : {int(y.sum())} | normales : {int((y == 0).sum())}")

    # 1. séparation entraînement / test
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2,
                                      stratify=y, random_state=SEED)
    Xtr, ytr, Xte, yte = X.iloc[idx_tr], y.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_te]
    print(f"Entraînement : {len(idx_tr)} | test (mis de côté) : {len(idx_te)}")

    # 2. comparaison par validation croisée (entraînement seulement)
    models = make_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, oof = [], {}
    for name, model in models.items():
        proba = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        oof[name] = proba
        rows.append({"modèle": name, **{k: v for k, v in metrics(ytr, proba).items()
                                         if k in ("accuracy", "sensibilité", "spécificité", "AUC")}})
    cvdf = pd.DataFrame(rows).set_index("modèle")
    print("\nValidation croisée 5 plis (sur l'entraînement) :")
    print(cvdf.round(3).to_string())
    if args.model == "auto":
        best = cvdf["AUC"].idxmax()
        print(f"\nMeilleur modèle (AUC en validation croisée) : {best}")
    elif args.model in models:
        best = args.model
        print(f"\nModèle choisi : {best}  (meilleur en validation croisée : {cvdf['AUC'].idxmax()})")
    else:
        raise SystemExit(f"Modèle inconnu : {args.model}. Choix : {list(models)} ou auto")

    # 3. évaluation finale sur le test
    final = clone(models[best]).fit(Xtr, ytr)
    p_te = final.predict_proba(Xte)[:, 1]
    m = metrics(yte, p_te)
    print("\nRésultats sur le jeu de TEST (jamais vu) :")
    for k in ("accuracy", "sensibilité", "spécificité", "précision", "F1", "AUC"):
        print(f"  {k:12s}: {m[k]:.3f}")
    print(f"  matrice de confusion : TN={m['TN']} FP={m['FP']} FN={m['FN']} TP={m['TP']}")

    # figures : ROC (validation croisée, 3 modèles + test) et matrice de confusion
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, proba in oof.items():
        fpr, tpr, _ = roc_curve(ytr, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={cvdf.loc[name, 'AUC']:.2f})")
    fpr, tpr, _ = roc_curve(yte, p_te)
    ax.plot(fpr, tpr, "k--", lw=2, label=f"Test : {best} (AUC={m['AUC']:.2f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set_xlabel("1 - spécificité")
    ax.set_ylabel("Sensibilité")
    ax.set_title("Courbes ROC")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "roc.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = confusion_matrix(yte, (p_te >= 0.5).astype(int), labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "TB"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Matrice de confusion (test)\n{best}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # 4. contrôle : entraîner sur un county, tester sur l'autre
    counties = [c for c in df["county"].dropna().unique()]
    cross_rows = []
    if len(counties) == 2:
        print("\nContrôle inter-hôpitaux (entraînement sur l'un, test sur l'autre) :")
        for a, b in ((counties[0], counties[1]), (counties[1], counties[0])):
            tr, te = df["county"] == a, df["county"] == b
            if y[tr].nunique() < 2 or y[te].nunique() < 2:
                continue
            mdl = clone(models[best]).fit(X[tr], y[tr])
            r = metrics(y[te], mdl.predict_proba(X[te])[:, 1])
            print(f"  train={a} ({int(tr.sum())}) -> test={b} ({int(te.sum())}) : "
                  f"AUC={r['AUC']:.3f}  sens={r['sensibilité']:.3f}  spéc={r['spécificité']:.3f}")
            cross_rows.append({"train": a, "test": b, **{k: r[k] for k in
                               ("AUC", "sensibilité", "spécificité", "accuracy")}})
    else:
        print("\n(contrôle inter-hôpitaux ignoré : il faut exactement 2 valeurs de 'county')")

    # importance des caractéristiques (Random Forest, sur l'entraînement)
    rf = clone(models["Random Forest"]).fit(Xtr, ytr)
    imp = pd.Series(rf[-1].feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\nCaractéristiques les plus utiles (Random Forest) :")
    print(imp.head(8).round(3).to_string())

    # 5. sauvegardes
    cvdf.to_csv(out / "classification_cv.csv")
    pd.DataFrame([{"modèle": best, **{k: m[k] for k in m}}]).to_csv(
        out / "classification_test.csv", index=False)
    if cross_rows:
        pd.DataFrame(cross_rows).to_csv(out / "classification_inter_county.csv", index=False)
    imp.to_csv(out / "feature_importance.csv", header=["importance"])
    deploy = clone(models[best]).fit(X, y)   # modèle final : toutes les données
    joblib.dump({"model": deploy, "features": feat_cols, "name": best},
                mdir / "tb_classifier.joblib")
    print(f"\nFichiers : {out}\\roc.png, confusion_matrix.png, classification_*.csv")
    print(f"Modèle : {mdir / 'tb_classifier.joblib'}")


if __name__ == "__main__":
    main()