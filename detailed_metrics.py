import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from novel_method import N_CNN, load_features
from train_model import load_cnn

NAMES = ["Healthy", "Caries", "Deep caries"]
COLORS = {"CNN only": "#9bb7d4", "Radiological only": "#d9822b", "Fusion (ours)": "#1f5fa8"}
plt.rcParams.update({"font.size": 12, "font.family": "serif"})


def pipeline(classifier, k):
    steps = [("scale", StandardScaler())]
    if k:
        steps.append(("select", SelectKBest(f_classif, k=k)))
    steps.append(("clf", clone(classifier)))
    return Pipeline(steps)


def specificity(y_true, y_pred, positive):
    tn = np.sum((y_true != positive) & (y_pred != positive))
    fp = np.sum((y_true != positive) & (y_pred == positive))
    return tn / (tn + fp)


def all_metrics(y, pred, prob):
    """3-class (macro average) and 2-class (healthy vs. carious) metrics."""
    y2, pred2, prob2 = (y > 0).astype(int), (pred > 0).astype(int), prob[:, 1] + prob[:, 2]
    return {
        "3-class": {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, average="macro", zero_division=0),
            "sensitivity": recall_score(y, pred, average="macro"),
            "specificity": float(np.mean([specificity(y, pred, c) for c in range(3)])),
            "f1": f1_score(y, pred, average="macro"),
            "auc": roc_auc_score(y, prob, multi_class="ovr", average="macro"),
            "kappa": cohen_kappa_score(y, pred),
            "mcc": matthews_corrcoef(y, pred),
        },
        "2-class": {
            "accuracy": accuracy_score(y2, pred2),
            "precision": precision_score(y2, pred2, zero_division=0),
            "sensitivity": recall_score(y2, pred2),
            "specificity": specificity(y2, pred2, 1),
            "f1": f1_score(y2, pred2),
            "auc": roc_auc_score(y2, prob2),
            "kappa": cohen_kappa_score(y2, pred2),
            "mcc": matthews_corrcoef(y2, pred2),
        },
    }


def per_class(y, pred, prob):
    rows = {}
    for c, name in enumerate(NAMES):
        rows[name] = {
            "precision": precision_score(y == c, pred == c, zero_division=0),
            "sensitivity": recall_score(y == c, pred == c),
            "specificity": specificity(y, pred, c),
            "f1": f1_score(y == c, pred == c),
            "auc": roc_auc_score(y == c, prob[:, c]),
        }
    return rows


def plot_confusion(ax, cm, labels, title):
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (dentist)")
    ax.set_title(title)


if __name__ == "__main__":
    start = time.time()
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Extracting features...")
    cnn = load_cnn()
    cnn_train, rad_train, y_train, rad_names = load_features(cnn, "train")
    cnn_test, rad_test, y_test, _ = load_features(cnn, "test")
    fus_train, fus_test = np.hstack([cnn_train, rad_train]), np.hstack([cnn_test, rad_test])

    # ---- models -------------------------------------------------------------------------
    print("Training models...")
    models = {
        "Decision Tree (CNN + FS)": (pipeline(DecisionTreeClassifier(random_state=42), 512), "cnn"),
        "k-NN (CNN + FS)": (pipeline(KNeighborsClassifier(n_neighbors=5), 512), "cnn"),
        "Random Forest (CNN)": (pipeline(RandomForestClassifier(n_estimators=300, random_state=42), None), "cnn"),
        "Logistic Regression (CNN)": (pipeline(LogisticRegression(max_iter=2000), None), "cnn"),
        "CNN only": (pipeline(SVC(kernel="rbf", C=1, probability=True, random_state=42), 512), "cnn"),
    }
    data = {"cnn": (cnn_train, cnn_test), "rad": (rad_train, rad_test), "fus": (fus_train, fus_test)}
    fitted = {name: model.fit(data[kind][0], y_train) for name, (model, kind) in models.items()}

    rad_search = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("select", SelectKBest(f_classif)),
                  ("clf", SVC(kernel="rbf", probability=True, random_state=42))]),
        {"select__k": [10, 20, 30, "all"], "clf__C": [1, 10]}, cv=5, scoring="f1_macro", n_jobs=1)
    fitted["Radiological only"] = rad_search.fit(rad_train, y_train).best_estimator_
    models["Radiological only"] = (None, "rad")
    fitted["Fusion (ours)"] = joblib.load(os.path.join("models", "fusion_model.joblib"))
    models["Fusion (ours)"] = (None, "fus")

    results, probs, preds = {}, {}, {}
    for name, model in fitted.items():
        X_test = data[models[name][1]][1]
        preds[name], probs[name] = model.predict(X_test), model.predict_proba(X_test)
        results[name] = all_metrics(y_test, preds[name], probs[name])
    ours = "Fusion (ours)"
    per_class_ours = per_class(y_test, preds[ours], probs[ours])

    # ---- tables -------------------------------------------------------------------------
    keys = ["accuracy", "precision", "sensitivity", "specificity", "f1", "auc", "kappa", "mcc"]
    head = "Model                       | Acc   Prec  Sens  Spec  F1    AUC   Kappa MCC"
    lines = []
    for task in ("3-class", "2-class"):
        lines += [f"{task.upper()} (macro average for 3 classes)" if task == "3-class"
                  else "2-CLASS (healthy vs. carious)", head, "-" * len(head)]
        for name in results:
            r = results[name][task]
            lines.append(f"{name:27s} | " + " ".join(f"{r[k]:.3f}" for k in keys))
        lines.append("")
    lines += ["NOVEL METHOD PER CLASS", "Class        | Prec  Sens  Spec  F1    AUC", "-" * 42]
    for name, r in per_class_ours.items():
        lines.append(f"{name:12s} | " + " ".join(f"{r[k]:.3f}" for k in
                                                 ["precision", "sensitivity", "specificity", "f1", "auc"]))
    cm3 = confusion_matrix(y_test, preds[ours])
    cm2 = confusion_matrix(y_test > 0, preds[ours] > 0)
    lines += ["", "Confusion matrix (3 classes, rows = true):", str(cm3),
              "Confusion matrix (2 classes, rows = true):", str(cm2)]

    # ---- graphs -------------------------------------------------------------------------
    print("Drawing graphs...")
    three = ["CNN only", "Radiological only", ours]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [3, 2]})
    plot_confusion(axes[0], cm3, NAMES, "3 classes")
    plot_confusion(axes[1], cm2, ["Healthy", "Carious"], "2 classes")
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "confusion_matrices.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name in three:
        fpr, tpr, _ = roc_curve(y_test > 0, probs[name][:, 1] + probs[name][:, 2])
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random guess")
    ax.set_xlabel("1 - specificity (false positive rate)")
    ax.set_ylabel("Sensitivity (true positive rate)")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "roc_2class.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for c, (name, color) in enumerate(zip(NAMES, ["#4c9a2a", "#e6a800", "#c0392b"])):
        fpr, tpr, _ = roc_curve(y_test == c, probs[ours][:, c])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random guess")
    ax.set_xlabel("1 - specificity (false positive rate)")
    ax.set_ylabel("Sensitivity (true positive rate)")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "roc_per_class.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name in three:
        prec, rec, _ = precision_recall_curve(y_test > 0, probs[name][:, 1] + probs[name][:, 2])
        ax.plot(rec, prec, color=COLORS[name], lw=2, label=name)
    ax.axhline((y_test > 0).mean(), color="k", ls=":", lw=1, label="Random guess")
    ax.set_xlabel("Sensitivity (recall)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "pr_2class.png"), dpi=200)

    radar_keys = ["accuracy", "precision", "sensitivity", "specificity", "f1", "auc"]
    labels = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1-score", "AUC"]
    angles = np.linspace(0, 2 * np.pi, len(radar_keys), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for name in three:
        values = [results[name]["3-class"][k] for k in radar_keys]
        ax.plot(angles + angles[:1], values + values[:1], color=COLORS[name], lw=2, label=name)
        ax.fill(angles + angles[:1], values + values[:1], color=COLORS[name], alpha=0.12)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_rlabel_position(90)
    lowest = min(results[n]["3-class"][k] for n in three for k in radar_keys)
    ax.set_ylim(max(0.0, round(lowest - 0.15, 1)), 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "radar_metrics.png"), dpi=200)

    select = fitted[ours].named_steps["select"]
    scores = np.nan_to_num(select.scores_)
    top = np.argsort(-scores)[:20][::-1]
    names = [f"CNN #{i}" if i < N_CNN else rad_names[i - N_CNN] for i in top]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh(range(20), scores[top], color=["#9bb7d4" if i < N_CNN else "#d9822b" for i in top])
    ax.set_yticks(range(20))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("ANOVA F-score")
    ax.bar(0, 0, color="#9bb7d4", label="CNN feature")
    ax.bar(0, 0, color="#d9822b", label="Radiological feature")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "top_features.png"), dpi=200)

    scaled = fitted[ours].named_steps["scale"].transform(fus_test)
    views = {"All 2048 CNN features": StandardScaler().fit(cnn_train).transform(cnn_test),
             "512 selected fusion features": select.transform(scaled)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (title, X) in zip(axes, views.items()):
        points = PCA(n_components=2, random_state=42).fit_transform(X)
        for c, (name, color) in enumerate(zip(NAMES, ["#4c9a2a", "#e6a800", "#c0392b"])):
            ax.scatter(*points[y_test == c].T, s=18, color=color, alpha=0.75, label=name)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "feature_space.png"), dpi=200)

    # ---- save ---------------------------------------------------------------------------
    print("\n" + "\n".join(lines))
    with open(os.path.join("results", "detailed_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join("results", "detailed_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"models": results, "per_class_fusion": per_class_ours,
                   "confusion_3class": cm3.tolist(), "confusion_2class": cm2.tolist()},
                  f, indent=2, default=float)
    print(f"\nSaved results/detailed_metrics.txt, results/detailed_metrics.json and 7 graphs in figures/"
          f"  ({time.time() - start:.0f}s)")