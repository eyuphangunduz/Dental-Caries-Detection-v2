import json
import os
import random
import time
import warnings

import numpy as np
from PIL import Image
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app import cnn, cut, find_teeth, tooth_features
from evaluate import compare, decide, overlap, to_corners
from novel_method import load_features

warnings.filterwarnings("ignore", category=FutureWarning)   # hide the scikit-learn 'probability' notice
random.seed(42)

SIZES = {   # number of non-carious training teeth (None = current healthy teeth from small_data)
    "A) 400 healthy (current)": None,
    "B) 400 non-carious": 400,
    "C) 1200 non-carious": 1200,
    "D) 2400 non-carious": 2400,
}
THRESHOLDS = [0.4, 0.5, 0.6]


def new_model():
    """The novel method with the settings chosen in novel_method.py (k=512, C=1)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=512)),
        ("svm", SVC(kernel="rbf", C=1, probability=True, random_state=42)),
    ])


def load_json(split):
    with open(os.path.join("data", f"{split}.json"), encoding="utf-8") as f:
        return json.load(f)


def non_carious_pool():
    """All teeth YOLO finds on the training X-rays that do not overlap a carious box (shuffled)."""
    images = load_json("train")
    pool, start = [], time.time()
    for n, info in enumerate(images, 1):
        carious = [to_corners(t["bbox"]) for t in info["teeth"] if t["label"] in ("caries", "deep_caries")]
        other = [to_corners(t["bbox"]) for t in info["teeth"] if t["label"] == "other"]
        for fdi, box in find_teeth(Image.open(info["path"]).convert("RGB")):
            if all(overlap(box, c) < 0.05 for c in carious):
                is_other = any(overlap(box, o) >= 0.3 for o in other)   # impacted / periapical lesion
                pool.append((info["path"], box, is_other))
        if n % 50 == 0 or n == len(images):
            print(f"  {n}/{len(images)} training X-rays  ({time.time() - start:.0f}s)")
    random.shuffle(pool)
    return pool


def pool_features(items):
    """Features of the chosen non-carious teeth (each X-ray is opened only once)."""
    features = [None] * len(items)
    by_xray = {}
    for index, (path, box, _) in enumerate(items):
        by_xray.setdefault(path, []).append((index, box))
    for path, teeth in by_xray.items():
        image = Image.open(path).convert("RGB")
        rows = tooth_features([cut(image, box) for _, box in teeth])
        for (index, _), row in zip(teeth, rows):
            features[index] = row
    return np.array(features)


def test_xrays():
    """YOLO teeth, their features and the dentists' carious teeth for the 113 test X-rays."""
    xrays = []
    for n, info in enumerate(load_json("test"), 1):
        image = Image.open(info["path"]).convert("RGB")
        teeth = find_teeth(image)
        X = tooth_features([cut(image, box) for fdi, box in teeth]) if teeth else None
        dentist = [(to_corners(t["bbox"]), t["label"]) for t in info["teeth"]
                   if t["label"] in ("caries", "deep_caries")]
        xrays.append((teeth, X, dentist))
        if n % 20 == 0:
            print(f"  {n} test X-rays")
    return xrays


if __name__ == "__main__":
    start = time.time()
    print("1/4 Features of the current tooth training and test sets...")
    cnn_tr, rad_tr, y_tr, _ = load_features(cnn, "train")
    cnn_te, rad_te, y_te, _ = load_features(cnn, "test")
    X_train, X_test = np.hstack([cnn_tr, rad_tr]), np.hstack([cnn_te, rad_te])
    carious_X, carious_y = X_train[y_tr > 0], y_tr[y_tr > 0]

    print("2/4 Finding non-carious teeth on the training X-rays (YOLO)...")
    pool = non_carious_pool()
    chosen = pool[:max(n for n in SIZES.values() if n)]
    print(f"  {len(pool)} non-carious teeth found, {sum(o for *_, o in pool)} of them "
          f"impacted / periapical lesion; using {len(chosen)}")

    print("3/4 Features of the chosen non-carious teeth...")
    extra_X = pool_features(chosen)

    print("4/4 Features of the 113 test X-rays (YOLO)...")
    xrays = test_xrays()
    total = sum(len(dentist) for _, _, dentist in xrays)

    lines = [f"Whole system: {len(xrays)} test X-rays, {total} carious teeth. "
             f"Tooth test: {len(y_te)} teeth.", "",
             "Healthy training teeth   | abnormal* | Tooth macro F1 | Threshold | Carious found | False alarms/X-ray",
             "-------------------------+-----------+----------------+-----------+---------------+-------------------"]
    for name, n in SIZES.items():
        healthy = X_train[y_tr == 0] if n is None else extra_X[:n]
        abnormal = 0 if n is None else sum(o for *_, o in chosen[:n])
        X = np.vstack([healthy, carious_X])
        y = np.concatenate([np.zeros(len(healthy), dtype=int), carious_y])
        model = new_model().fit(X, y)
        tooth_f1 = f1_score(y_te, model.predict(X_test), average="macro")
        probs = [model.predict_proba(features) if features is not None else [] for _, features, _ in xrays]
        for threshold in THRESHOLDS:
            found = alarms = 0
            for (teeth, _, dentist), p in zip(xrays, probs):
                f, _, a = compare(teeth, decide(p, threshold), dentist)
                found, alarms = found + f, alarms + a
            lines.append(f"{name:24s} | {abnormal:9d} |     {tooth_f1:.3f}      |    {threshold:.1f}    |"
                         f"  {100 * found / total:4.0f}%        |  {alarms / len(xrays):5.2f}")
        lines.append("")
        print(f"  trained {name}")

    lines.append("* abnormal = non-carious training teeth that DENTEX marks as impacted or periapical lesion.")
    print("\n" + "\n".join(lines))
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "more_data.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved results/more_data.txt  ({(time.time() - start) / 60:.0f} min)")