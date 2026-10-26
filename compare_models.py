import os
import time

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from train_model import load_cnn, load_split

K = 512

MODELS = {
    "SVM (ours)": SVC(kernel="rbf", C=1),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}


def make_pipeline(classifier, k):
    steps = [("scale", StandardScaler())]
    if k:
        steps.append(("select", SelectKBest(f_classif, k=k)))
    steps.append(("classifier", clone(classifier)))
    return Pipeline(steps)


def carious(labels):
    labels = np.asarray(labels)
    healthy = 0 if np.issubdtype(labels.dtype, np.number) else "healthy"
    return (labels != healthy).astype(int)


if __name__ == "__main__":
    start = time.time()
    print("Extracting CNN features...")
    cnn = load_cnn()
    X_train, y_train = load_split(cnn, "train")
    X_test, y_test = load_split(cnn, "test")
    print(f"Train: {len(y_train)} teeth, test: {len(y_test)} teeth\n")

    lines = ["                      |          |     3 classes     |        2 classes",
             "Model                 | Features | Accuracy Macro F1 | Accuracy  F1   Recall",
             "----------------------+----------+-------------------+---------------------"]
    for name, classifier in MODELS.items():
        for k in (None, K):
            model = make_pipeline(classifier, k).fit(X_train, y_train)
            pred = model.predict(X_test)
            acc3 = accuracy_score(y_test, pred)
            f1_3 = f1_score(y_test, pred, average="macro")
            true2, pred2 = carious(y_test), carious(pred)
            acc2 = accuracy_score(true2, pred2)
            f1_2 = f1_score(true2, pred2)
            rec2 = recall_score(true2, pred2)
            features = f"{k} (FS)" if k else "2048"
            lines.append(f"{name:21s} | {features:8s} |  {acc3:.3f}    {f1_3:.3f}   |"
                         f"  {acc2:.3f}   {f1_2:.3f}  {rec2:.3f}")
        lines.append("")

    lines.append("FS = ANOVA feature selection. 2 classes: caries and deep caries counted together as 'caries'.")
    print("\n".join(lines))
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "model_comparison.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved to results/model_comparison.txt  ({time.time() - start:.0f}s)")