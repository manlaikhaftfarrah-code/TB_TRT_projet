"""
features.py : extraction de caractéristiques dans les poumons segmentés.

Pour chaque radiographie :
  1. on segmente les poumons avec NOTRE code (segmentation.py),
  2. on calcule des caractéristiques uniquement à l'intérieur du masque,
  3. on enregistre tout dans results/features.csv (une ligne par image).

Utilisation (depuis le dossier du projet, environnement activé) :
    python features.py --n 20       # test rapide sur 20 images
    python features.py              # les 704 images (quelques minutes)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from skimage import measure
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from segmentation import (BASE, DEFAULT_ROOT, dice, image_dir, load_gray,
                          load_mask, mask_dir, segment_lungs)

LEVELS = 32  # niveaux de gris pour la GLCM


def _mean(img, m):
    return float(img[m].mean()) if m.any() else np.nan


def split_lungs(mask):
    """Retourne (poumon gauche, poumon droit) *de l'image*, par position horizontale.

    Si les poumons forment une seule région (collés), on coupe le masque à la
    médiane horizontale de ses pixels.
    """
    lab = measure.label(mask)
    props = sorted(measure.regionprops(lab), key=lambda p: p.area, reverse=True)[:2]
    if len(props) >= 2:
        props = sorted(props, key=lambda p: p.centroid[1])
        return lab == props[0].label, lab == props[1].label
    cut = int(np.median(np.nonzero(mask)[1]))
    cols = np.arange(mask.shape[1])[None, :]
    return mask & (cols < cut), mask & (cols >= cut)


def extract_features(enh, mask):
    """Dictionnaire de caractéristiques calculées dans `mask` sur l'image `enh` (CLAHE)."""
    f = {}
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    left, right = split_lungs(mask)

    # ---- Forme ----
    a_l, a_r = int(left.sum()), int(right.sum())
    f["n_regions"] = int(measure.label(mask).max())
    f["area_frac"] = mask.sum() / mask.size
    f["area_asym"] = abs(a_l - a_r) / (a_l + a_r) if (a_l + a_r) else np.nan
    f["bbox_aspect"] = (y1 - y0 + 1) / (x1 - x0 + 1)
    f["solidity"] = measure.regionprops(mask.astype(np.uint8))[0].solidity

    # ---- Intensité dans les poumons ----
    v = enh[mask].astype(np.float64)
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    f.update(int_mean=v.mean(), int_std=v.std(), int_skew=float(stats.skew(v)),
             int_kurt=float(stats.kurtosis(v)), int_p10=p10, int_p50=p50, int_p90=p90)

    # ---- Intensité par rapport au reste du corps (limite l'effet de l'exposition) ----
    rest = enh[(enh > 15) & ~mask]
    if rest.size > 50:
        f["lung_vs_body"] = v.mean() / max(rest.mean(), 1e-6)
        f["frac_bright"] = float((v > np.percentile(rest, 25)).mean())
    else:
        f["lung_vs_body"] = f["frac_bright"] = np.nan

    # ---- Zones haute / moyenne / basse (la TB touche souvent les sommets) ----
    edges = np.linspace(y0, y1 + 1, 4).astype(int)
    zone_means = []
    for name, lo, hi in zip(("up", "mid", "low"), edges[:-1], edges[1:]):
        z = mask.copy()
        z[:lo] = False
        z[hi:] = False
        vals = enh[z]
        f[f"zone_{name}_mean"] = float(vals.mean()) if vals.size else np.nan
        f[f"zone_{name}_std"] = float(vals.std()) if vals.size else np.nan
        zone_means.append(f[f"zone_{name}_mean"])
    f["zone_up_minus_low"] = zone_means[0] - zone_means[2]

    # ---- Asymétrie gauche/droite ----
    ml, mr = _mean(enh, left), _mean(enh, right)
    f["lr_mean_diff"] = abs(ml - mr) if not (np.isnan(ml) or np.isnan(mr)) else np.nan

    # ---- Texture GLCM (le fond hors poumon est exclu : niveau 0 ignoré) ----
    crop = enh[y0:y1 + 1, x0:x1 + 1].astype(np.int32)
    mcrop = mask[y0:y1 + 1, x0:x1 + 1]
    q = 1 + crop * (LEVELS - 2) // 255
    q[~mcrop] = 0
    glcm = graycomatrix(q.astype(np.uint8), distances=[1, 3],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=LEVELS, symmetric=True, normed=False).astype(np.float64)
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    sums = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = glcm / np.where(sums == 0, 1, sums)
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        f[f"glcm_{prop}"] = float(np.nan_to_num(graycoprops(glcm, prop).mean()))

    # ---- Texture LBP (histogramme de 10 motifs "uniformes") ----
    lbp = local_binary_pattern(enh, 8, 1, method="uniform")
    hist = np.bincount(lbp[mask].astype(int), minlength=10)[:10].astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    for i, h in enumerate(hist):
        f[f"lbp_{i}"] = float(h)
    return f


def build_features(root, method="otsu", n=None, seed=0):
    meta = pd.read_csv(Path(root) / "MetaData.csv")
    if n:
        meta = meta.sample(min(n, len(meta)), random_state=seed)
    rows, skipped = [], 0
    total = len(meta)
    for k, (_, r) in enumerate(meta.iterrows(), 1):
        img_id = int(r["id"])
        ip = image_dir(root) / f"{img_id}.png"
        if not ip.exists():
            skipped += 1
            continue
        img = load_gray(ip)
        mask, st = segment_lungs(img, method)
        if not mask.any():
            print(f"  (segmentation vide pour l'id {img_id}, ignorée)")
            skipped += 1
            continue
        feats = extract_features(st["clahe"], mask)
        feats.update(id=img_id, ptb=int(r["ptb"]), county=r.get("county", ""),
                     erosion=st["erosion"])
        mp = mask_dir(root) / f"{img_id}.png"
        feats["dice_seg"] = dice(mask, load_mask(mp)) if mp.exists() else np.nan
        rows.append(feats)
        if k % 25 == 0 or k == total:
            print(f"  {k}/{total} images traitées")
    df = pd.DataFrame(rows)
    first = ["id", "ptb", "county", "erosion", "dice_seg"]
    return df[first + [c for c in df.columns if c not in first]], skipped


def main():
    ap = argparse.ArgumentParser(description="Extraction de caractéristiques")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="dossier tb_dataset")
    ap.add_argument("--n", type=int, default=0, help="nombre d'images (0 = toutes)")
    ap.add_argument("--method", default="otsu", choices=("otsu", "body_otsu"))
    ap.add_argument("--out", default=str(BASE / "results"), help="dossier de sortie")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, skipped = build_features(Path(args.root), args.method, args.n or None)
    if df.empty:
        raise SystemExit("Aucune caractéristique extraite : vérifie le chemin --root.")
    path = out / "features.csv"
    df.to_csv(path, index=False)

    n_feat = df.shape[1] - 5
    print(f"\nImages : {len(df)} (ignorées : {skipped}) | caractéristiques : {n_feat}")
    print(f"TB : {int(df.ptb.sum())}  |  normales : {int((df.ptb == 0).sum())}")
    print(f"Valeurs manquantes : {int(df.isna().sum().sum())}")
    print(f"Dice moyen de la segmentation : {df.dice_seg.mean():.3f}")
    show = ["area_frac", "int_mean", "lung_vs_body", "zone_up_minus_low", "glcm_contrast"]
    print("\nMoyennes par classe (ptb 0 = normal, 1 = TB) :")
    print(df.groupby("ptb")[show].mean().round(3).to_string())
    print(f"\nFichier : {path}")


if __name__ == "__main__":
    main()