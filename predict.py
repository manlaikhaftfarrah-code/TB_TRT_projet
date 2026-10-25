"""
predict.py : analyse d'une radiographie avec le modèle entraîné (sans interface).

    segmentation des poumons -> caractéristiques -> score de suspicion TB

Utilisé par app.py ; peut aussi s'utiliser seul :
    from predict import load_bundle, analyse
"""
import cv2
import joblib
import numpy as np
import pandas as pd

from features import extract_features
from segmentation import BASE, SIZE, _plausible, segment_lungs

MODEL_PATH = BASE / "models" / "tb_classifier.joblib"


def load_bundle(path=MODEL_PATH):
    """Charge le modèle sauvegardé par classifier.py."""
    return joblib.load(path)


def read_image_bytes(data):
    """Décode une image (octets d'un fichier PNG/JPG) en niveaux de gris SIZE x SIZE.

    Retourne None si le fichier n'est pas une image lisible.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def analyse(img, bundle, method="otsu"):
    """Segmente les poumons et calcule le score de suspicion TB.

    Retourne un dictionnaire :
        mask      masque booléen des poumons (peut être vide)
        steps     étapes intermédiaires (original, clahe, seuil, ...)
        plausible True si la segmentation a une forme/surface réaliste
        features  DataFrame d'une ligne (les caractéristiques du modèle) ou None
        proba     probabilité TB estimée (0 à 1) ou None si la segmentation a échoué
    """
    mask, steps = segment_lungs(img, method)
    out = {"mask": mask, "steps": steps, "plausible": _plausible(mask),
           "features": None, "proba": None}
    if not mask.any():
        return out
    feats = extract_features(steps["clahe"], mask)
    X = pd.DataFrame([feats]).reindex(columns=bundle["features"])
    out["features"] = X
    out["proba"] = float(bundle["model"].predict_proba(X)[0, 1])
    return out