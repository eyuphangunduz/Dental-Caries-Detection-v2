import os
import time

import joblib
import numpy as np
from PIL import Image
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_model import CLASSES, DATA, images_to_features, load_cnn

N_CNN = 2048   # number of ResNet-50 features


def radiological_features(image):
    """Handcrafted features of one tooth image -> dict {name: value}."""
    gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.uint8)
    pixels = gray.ravel() / 255.0
    f = {}

    # 1) Intensity: caries make parts of the tooth darker
    f["mean"] = pixels.mean()
    f["std"] = pixels.std()
    f["skewness"] = skew(pixels)
    f["kurtosis"] = kurtosis(pixels)
    for p in (10, 25, 50, 75, 90):
        f[f"percentile_{p}"] = np.percentile(pixels, p)
    median = np.median(pixels)
    f["dark_ratio_70"] = (pixels < 0.7 * median).mean()   # share of pixels much darker than the tooth
    f["dark_ratio_50"] = (pixels < 0.5 * median).mean()
    hist, _ = np.histogram(pixels, bins=16, range=(0, 1))
    hist = hist / hist.sum()
    for i, h in enumerate(hist):
        f[f"hist_{i}"] = h
    f["entropy"] = -(hist[hist > 0] * np.log2(hist[hist > 0])).sum()

    # 2) GLCM texture (as in the base paper), averaged over 4 directions
    levels = (gray // 8).astype(np.uint8)   # 256 -> 32 grey levels
    glcm = graycomatrix(levels, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        values = graycoprops(glcm, prop)   # rows = distances, columns = directions
        for row, distance in enumerate((1, 3)):
            f[f"glcm_{prop}_d{distance}"] = values[row].mean()

    # 3) LBP: small local patterns (edges, spots, flat areas)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    lbp_hist = lbp_hist / lbp_hist.sum()
    for i, h in enumerate(lbp_hist):
        f[f"lbp_{i}"] = h

    return {name: float(np.nan_to_num(value)) for name, value in f.items()}


def load_features(cnn, split):
    """Returns CNN features, handcrafted features, labels and handcrafted feature names."""
    cnn_parts, rad_rows, labels, names = [], [], [], None
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATA, split, cls)
        files = sorted(os.listdir(folder))
        for i in range(0, len(files), 32):
            images = [Image.open(os.path.join(folder, name)).convert("RGB") for name in files[i:i + 32]]
            cnn_parts.append(images_to_features(cnn, images))
            for image in images:
                features = radiological_features(image)
                names = list(features)
                rad_rows.append(list(features.values()))
        labels += [label] * len(files)
        print(f"  {split}/{cls}: {len(files)} images")
    return np.vstack(cnn_parts), np.array(rad_rows), np.array(labels), names


def train(X_train, y_train, k_values):
    """Scaling -> ANOVA -> SVM, k and C chosen by 5-fold cross-validation (as in train_model.py)."""
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif)),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
    ])
    search = GridSearchCV(pipeline, {"select__k": k_values, "svm__C": [1, 10]},
                          cv=5, scoring="f1_macro", n_jobs=-1)
    return search.fit(X_train, y_train)


def scores(model, X_test, y_test):
    pred = model.predict(X_test)
    return (accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro"),
            f1_score(y_test > 0, pred > 0))   # 2 classes: healthy (0) vs any caries


if __name__ == "__main__":
    start = time.time()
    print("Extracting features...")
    cnn = load_cnn()
    cnn_train, rad_train, y_train, rad_names = load_features(cnn, "train")
    cnn_test, rad_test, y_test, _ = load_features(cnn, "test")
    n_rad = rad_train.shape[1]
    print(f"{N_CNN} CNN features + {n_rad} handcrafted features per tooth\n")

    experiments = {
        "A) CNN only (previous)": (cnn_train, cnn_test, [32, 64, 128, 256, 512]),
        "B) Handcrafted only (base paper)": (rad_train, rad_test, [10, 20, 30, "all"]),
        "C) CNN + handcrafted (novel)": (np.hstack([cnn_train, rad_train]),
                                         np.hstack([cnn_test, rad_test]), [32, 64, 128, 256, 512, 1024]),
    }
    lines = ["Model                            | Features | k    C  | 3-class Acc  Macro F1 | 2-class F1",
             "---------------------------------+----------+---------+-----------------------+-----------"]
    models = {}
    for name, (X_train, X_test, k_values) in experiments.items():
        models[name] = model = train(X_train, y_train, k_values)
        acc, f1_3, f1_2 = scores(model, X_test, y_test)
        k, c = model.best_params_["select__k"], model.best_params_["svm__C"]
        lines.append(f"{name:32s} | {X_train.shape[1]:8d} | {str(k):4s} {c:2d} |"
                     f"    {acc:.3f}      {f1_3:.3f}   |   {f1_2:.3f}")

    # Which features did the novel model choose?
    fusion = models["C) CNN + handcrafted (novel)"].best_estimator_
    select = fusion.named_steps["select"]
    chosen = select.get_support()
    ranking = np.argsort(-np.nan_to_num(select.scores_))          # all features, best first
    position = {index: place + 1 for place, index in enumerate(ranking)}
    best_rad = sorted(range(n_rad), key=lambda i: position[N_CNN + i])[:10]
    lines += ["",
              f"Novel model kept {chosen.sum()} features: {chosen[:N_CNN].sum()} CNN + "
              f"{chosen[N_CNN:].sum()} handcrafted (of {n_rad}).",
              f"Best 10 handcrafted features (rank among all {N_CNN + n_rad} features by ANOVA F-score):"]
    lines += [f"  #{position[N_CNN + i]:4d}  {rad_names[i]}" for i in best_rad]

    print("\n".join(lines))
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "novel_method.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    joblib.dump(fusion, os.path.join("models", "fusion_model.joblib"))
    print(f"\nSaved results/novel_method.txt and models/fusion_model.joblib  ({time.time() - start:.0f}s)")