"""
segmentation.py : segmentation classique des poumons sur radiographie thoracique.

Pipeline : niveaux de gris -> redimensionnement -> flou gaussien -> CLAHE
           -> seuillage (Otsu, ou Otsu restreint au corps) -> morphologie
           -> suppression des régions touchant le bord -> 2 plus grandes régions.

Utilisation (depuis le dossier du projet, environnement activé) :
    python segmentation.py                # évalue 40 images au hasard + figure de démo
    python segmentation.py --n 100        # évalue 100 images
    python segmentation.py --all          # évalue les 704 images
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # on enregistre les figures dans des fichiers
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation

SIZE = 512
BASE = Path(__file__).resolve().parent
DEFAULT_ROOT = BASE / "data" / "tb_dataset"
METHODS = ("otsu", "body_otsu")


# --------------------------------------------------------------------------
# Chemins et chargement
# --------------------------------------------------------------------------
def image_dir(root):
    return Path(root) / "Chest-X-Ray" / "Chest-X-Ray" / "image"


def mask_dir(root):
    return Path(root) / "Chest-X-Ray" / "Chest-X-Ray" / "mask"


def load_gray(path, size=SIZE):
    """Lit une image en niveaux de gris 8 bits et la redimensionne."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image illisible ou introuvable : {path}")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_mask(path, size=SIZE):
    """Lit un masque de référence et le convertit en booléen."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Masque illisible ou introuvable : {path}")
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return m > 127


# --------------------------------------------------------------------------
# Étapes du pipeline
# --------------------------------------------------------------------------
def preprocess(img):
    """Débruitage léger (gaussien) puis amélioration du contraste (CLAHE)."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(blur)


def body_mask(enh):
    """Masque du corps : plus grande région non noire, trous remplis (poumons inclus)."""
    lab = measure.label(enh > 15)
    if lab.max() == 0:
        return np.ones_like(enh, dtype=bool)
    sizes = np.bincount(lab.ravel())[1:]
    body = lab == (np.argmax(sizes) + 1)
    return ndi.binary_fill_holes(body)


def lung_threshold(enh, method="otsu"):
    """Seuil séparant les poumons (sombres) du reste du thorax.

    - "otsu"      : Otsu sur la zone centrale de l'image (noir pur ignoré).
    - "body_otsu" : Otsu calculé seulement à l'intérieur du corps, donc le fond
                    de l'image n'influence pas le seuil.
    """
    h, w = enh.shape
    if method == "body_otsu":
        vals = enh[body_mask(enh)]
    else:
        roi = enh[int(0.10 * h): int(0.95 * h), int(0.10 * w): int(0.90 * w)]
        vals = roi[roi > 5]
    if vals.size < 100:
        return 128.0
    return float(filters.threshold_otsu(vals))


def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def clean_mask(cand, min_area_frac=0.01, n_keep=2, erode=0):
    """Nettoie le masque candidat et ne garde que les deux plus grandes régions.

    erode > 0 : on érode d'abord le masque (rayon en pixels) pour couper les
    "ponts" fins qui relient les poumons au fond ; on supprime ensuite les
    régions touchant le bord, puis on redilate (limité au masque d'origine).
    """
    m = cand.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(5))    # enlève les petits points
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(9))   # bouche les petits trous
    k = _ellipse(2 * erode + 1) if erode > 0 else None
    core = cv2.erode(m, k) if erode > 0 else m
    core = segmentation.clear_border(core.astype(bool))     # enlève le fond (touche le bord)
    lab = measure.label(core)
    min_area = min_area_frac * core.size * (0.3 if erode > 0 else 1.0)
    props = [p for p in measure.regionprops(lab) if p.area >= min_area]
    props = sorted(props, key=lambda p: p.area, reverse=True)[:n_keep]
    out = np.zeros(core.shape, dtype=bool)
    for p in props:
        out |= lab == p.label
    if erode > 0:
        out = cv2.dilate(out.astype(np.uint8), k).astype(bool) & m.astype(bool)
    return ndi.binary_fill_holes(out)


def _plausible(mask):
    """Deux régions et une surface réaliste (8 % à 45 % de l'image)."""
    n = measure.label(mask).max()
    return n >= 2 and 0.08 <= mask.mean() <= 0.45


def segment_lungs(img, method="otsu"):
    """Segmente les poumons. Retourne (masque booléen, dict des étapes).

    On essaie d'abord sans érosion ; si le résultat n'est pas plausible
    (poumons collés au fond, masque vide...), on réessaie avec une érosion
    de plus en plus forte.
    """
    enh = preprocess(img)
    t = lung_threshold(enh, method)
    cand = enh < t
    tried = []
    for r in (0, 6, 12, 18, 24):
        mask = clean_mask(cand, erode=r)
        tried.append((r, mask))
        if _plausible(mask):
            break
    else:  # aucun essai plausible : on garde le plus grand masque raisonnable
        ok = [(r, m) for r, m in tried if m.mean() <= 0.45 and m.any()]
        r, mask = max(ok, key=lambda x: x[1].sum()) if ok else tried[-1]
    steps = {"original": img, "clahe": enh, "seuil": cand, "masque": mask, "t": t, "erosion": r}
    return mask, steps


# --------------------------------------------------------------------------
# Métriques
# --------------------------------------------------------------------------
def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else np.logical_and(a, b).sum() / u


# --------------------------------------------------------------------------
# Affichage
# --------------------------------------------------------------------------
def overlay(img, pred, gt=None):
    """Contours : vert = notre segmentation, rouge = masque de référence."""
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for m, color in ((gt, (255, 0, 0)), (pred, (0, 255, 0))):
        if m is None:
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cnts, -1, color, 2)
    return rgb


def save_demo_figure(root, ids, method, out_path):
    """Figure : pour chaque image, original / CLAHE / seuillage / résultat."""
    fig, axes = plt.subplots(len(ids), 4, figsize=(14, 3.6 * len(ids)), squeeze=False)
    for row, (img_id, label) in enumerate(ids):
        img = load_gray(image_dir(root) / f"{img_id}.png")
        gt = load_mask(mask_dir(root) / f"{img_id}.png")
        pred, st = segment_lungs(img, method)
        d = dice(pred, gt)
        panels = [
            (f"{img_id} ({label}) : original", st["original"], "gray"),
            ("CLAHE", st["clahe"], "gray"),
            (f"Seuillage ({method}, t={st['t']:.0f}, érosion {st['erosion']})", st["seuil"], "gray"),
            (f"Résultat (vert) vs référence (rouge) : Dice={d:.2f}", overlay(img, pred, gt), None),
        ]
        for ax, (title, im, cmap) in zip(axes[row], panels):
            ax.imshow(im, cmap=cmap) if cmap else ax.imshow(im)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.suptitle("Segmentation des poumons : méthode classique", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# Évaluation sur le dataset
# --------------------------------------------------------------------------
def evaluate(root, n=40, seed=0):
    meta = pd.read_csv(Path(root) / "MetaData.csv")
    if n:
        meta = meta.sample(min(n, len(meta)), random_state=seed)
    rows = []
    for _, r in meta.iterrows():
        img_id = int(r["id"])
        ip = image_dir(root) / f"{img_id}.png"
        mp = mask_dir(root) / f"{img_id}.png"
        if not (ip.exists() and mp.exists()):
            print(f"  (ignorée : fichier manquant pour l'id {img_id})")
            continue
        img = load_gray(ip)
        gt = load_mask(mp)
        row = {"id": img_id, "ptb": int(r["ptb"]), "county": r.get("county", "")}
        for m in METHODS:
            pred, _ = segment_lungs(img, m)
            row[f"dice_{m}"] = dice(pred, gt)
            row[f"iou_{m}"] = iou(pred, gt)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Segmentation classique des poumons")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="dossier tb_dataset")
    ap.add_argument("--n", type=int, default=40, help="nombre d'images à évaluer")
    ap.add_argument("--all", action="store_true", help="évaluer toutes les images")
    ap.add_argument("--out", default=str(BASE / "results"), help="dossier de sortie")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = evaluate(root, None if args.all else args.n)
    if df.empty:
        raise SystemExit("Aucune image évaluée : vérifie le chemin --root.")
    df.to_csv(out / "segmentation_eval.csv", index=False)

    print(f"\nImages évaluées : {len(df)}")
    cols = [c for c in df.columns if c.startswith(("dice_", "iou_"))]
    print("\nMoyennes globales :")
    print(df[cols].mean().round(3).to_string())
    print("\nMoyennes par classe (ptb 0 = normal, 1 = TB) :")
    print(df.groupby("ptb")[cols].mean().round(3).to_string())

    best = max(METHODS, key=lambda m: df[f"dice_{m}"].mean())
    print(f"\nMeilleure méthode (Dice moyen) : {best}")

    # figure de démo : 2 normales + 2 TB (si disponibles)
    ids = []
    for ptb, label in ((0, "normal"), (1, "TB")):
        sub = df[df.ptb == ptb].head(2)
        ids += [(int(i), label) for i in sub["id"]]
    save_demo_figure(root, ids, best, out / "segmentation_demo.png")
    print(f"\nFigure : {out / 'segmentation_demo.png'}")
    print(f"Tableau : {out / 'segmentation_eval.csv'}")


if __name__ == "__main__":
    main()