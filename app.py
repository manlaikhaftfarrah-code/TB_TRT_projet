"""
app.py : interface Streamlit.

Lancer (depuis le dossier du projet, environnement activé) :
    streamlit run app.py

Prérequis : python features.py puis python classifier.py (crée le modèle).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from predict import MODEL_PATH, analyse, load_bundle, read_image_bytes
from segmentation import DEFAULT_ROOT, dice, image_dir, load_gray, load_mask, mask_dir, overlay

# (les variables d'environnement servent uniquement aux tests automatiques)
ROOT = Path(os.environ.get("TB_DATA_ROOT", DEFAULT_ROOT))
MODEL = Path(os.environ.get("TB_MODEL", MODEL_PATH))
RESULTS = MODEL.parent.parent / "results"

st.set_page_config(page_title="Dépistage TB sur radiographie", layout="wide")


@st.cache_resource
def get_bundle(path):
    return load_bundle(path)


@st.cache_data
def get_meta(root):
    p = Path(root) / "MetaData.csv"
    return pd.read_csv(p) if p.exists() else None


st.title("Segmentation des poumons et aide au dépistage de la tuberculose")
st.warning(
    "Outil pédagogique. Il ne pose AUCUN diagnostic et ne remplace pas l'avis d'un médecin. "
    "Les résultats ne sont pas fiables pour un usage clinique."
)

if not MODEL.exists():
    st.error("Modèle introuvable. Lance d'abord : python features.py, puis python classifier.py")
    st.stop()
bundle = get_bundle(str(MODEL))

# ---------------------------------------------------------------- barre latérale
with st.sidebar:
    st.header("Paramètres")
    source = st.radio("Source de l'image", ["Importer une radiographie", "Exemple du dataset"])
    method = st.selectbox("Méthode de seuillage", ["otsu", "body_otsu"])
    thr = st.slider("Seuil de décision (score ≥ seuil : suspect)", 0.10, 0.90, 0.50, 0.05)
    st.caption(
        "Baisser le seuil détecte plus de cas de TB (meilleure sensibilité) mais donne plus de "
        "fausses alertes (spécificité plus faible)."
    )

# ---------------------------------------------------------------- choix de l'image
img, gt, truth = None, None, None
if source == "Importer une radiographie":
    up = st.file_uploader("Radiographie thoracique de face (PNG ou JPG)", type=["png", "jpg", "jpeg"])
    if up is not None:
        img = read_image_bytes(up.getvalue())
        if img is None:
            st.error("Fichier illisible : choisis une image PNG ou JPG.")
            st.stop()
else:
    meta = get_meta(str(ROOT))
    if meta is None:
        st.sidebar.error(f"Dataset introuvable : {ROOT}")
        st.stop()
    ids = meta["id"].astype(int).tolist()
    chosen = st.sidebar.selectbox("Image du dataset (id)", ids)
    ip, mp = image_dir(ROOT) / f"{chosen}.png", mask_dir(ROOT) / f"{chosen}.png"
    if not ip.exists():
        st.error(f"Image introuvable : {ip}")
        st.stop()
    img = load_gray(ip)
    gt = load_mask(mp) if mp.exists() else None
    truth = int(meta.loc[meta["id"].astype(int) == chosen, "ptb"].iloc[0])

if img is None:
    st.info("Importe une radiographie, ou choisis « Exemple du dataset » dans la barre latérale.")
    st.stop()

# ---------------------------------------------------------------- analyse
res = analyse(img, bundle, method)
mask, steps = res["mask"], res["steps"]

st.subheader("1. Segmentation des poumons")
c1, c2, c3, c4 = st.columns(4)
c1.image(img, caption="Radiographie (redimensionnée)")
c2.image(steps["clahe"], caption="Après flou et CLAHE")
c3.image((steps["seuil"] * 255).astype(np.uint8), caption=f"Seuillage ({method}, t = {steps['t']:.0f})")
c4.image(overlay(img, mask, gt), caption="Poumons trouvés (vert)" + (" / référence (rouge)" if gt is not None else ""))

if not mask.any():
    st.error("La segmentation a échoué (aucun poumon trouvé). Aucun score n'est calculé.")
    st.stop()
if not res["plausible"]:
    st.warning("Segmentation douteuse (nombre de régions ou surface inhabituels) : le score est peu fiable.")
if steps["erosion"] > 0:
    st.caption(f"Une érosion de {steps['erosion']} px a été nécessaire pour séparer les poumons du fond.")

# ---------------------------------------------------------------- score
st.subheader("2. Score de suspicion de tuberculose")
p = res["proba"]
m1, m2, m3 = st.columns(3)
m1.metric("Score de suspicion", f"{p:.0%}")
m2.metric("Seuil de décision", f"{thr:.0%}")
m3.metric("Surface des poumons", f"{mask.mean():.0%} de l'image")
st.progress(float(min(max(p, 0.0), 1.0)))

if p >= thr:
    st.error(f"Suspect : score {p:.0%} ≥ seuil {thr:.0%}. À faire examiner par un médecin.")
else:
    st.success(f"Non suspect selon le modèle : score {p:.0%} < seuil {thr:.0%}. Cela n'exclut pas la maladie.")

if truth is not None:
    txt = f"Vérité terrain du dataset : {'TB' if truth == 1 else 'normal'}"
    if gt is not None:
        txt += f" | Dice de la segmentation : {dice(mask, gt):.2f}"
    st.caption(txt)

with st.expander("Caractéristiques calculées dans les poumons"):
    st.dataframe(res["features"].T.rename(columns={0: "valeur"}))

with st.expander("Limites de cet outil"):
    test_csv = RESULTS / "classification_test.csv"
    if test_csv.exists():
        t = pd.read_csv(test_csv).iloc[0]
        st.write(
            f"Sur le jeu de test du projet : sensibilité {t['sensibilité']:.2f}, "
            f"spécificité {t['spécificité']:.2f}, AUC {t['AUC']:.2f}."
        )
    st.markdown(
        "- Ces performances sont trop faibles pour un usage médical réel.\n"
        "- Le modèle a été entraîné sur 2 hôpitaux seulement ; testé sur un autre hôpital, "
        "il se trompe davantage et son seuil de décision n'est plus adapté.\n"
        "- La segmentation classique échoue sur certaines radiographies "
        "(mauvaise exposition, poumons très opacifiés).\n"
        "- Un score élevé n'est pas un diagnostic et un score faible n'exclut pas la maladie."
    )