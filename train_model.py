import os
import time

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import models, transforms

CLASSES = ["healthy", "caries", "deep_caries"]
DATA = "small_data"

# ---------------------------------------------------------------
# PART 1: CNN that turns a tooth image into 2048 numbers (features)
# ---------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),                  # ResNet expects 224x224 images
    transforms.Grayscale(num_output_channels=3),    # X-ray has 1 channel, ResNet wants 3
    transforms.ToTensor(),                          # image -> numbers between 0 and 1
    transforms.Normalize([0.485, 0.456, 0.406],     # same normalisation as ImageNet
                         [0.229, 0.224, 0.225]),
])


def load_cnn():
    """Pretrained ResNet-50 (trained on ImageNet). We do NOT train it,
    we only use it to turn images into features."""
    cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    cnn.fc = torch.nn.Identity()    # remove the last layer -> output is 2048 features
    cnn.eval()                      # evaluation mode (no training)
    return cnn


def images_to_features(cnn, images):
    """List of PIL images -> numpy array of shape (number of images, 2048)."""
    batch = torch.stack([TRANSFORM(img) for img in images])
    with torch.no_grad():           # no training -> faster, less memory
        return cnn(batch).numpy()


# ---------------------------------------------------------------
# PART 2: Load all tooth images of a split and turn them into features
# ---------------------------------------------------------------
def load_split(cnn, split):
    X, y = [], []
    for label, class_name in enumerate(CLASSES):      # healthy=0, caries=1, deep_caries=2
        folder = os.path.join(DATA, split, class_name)
        files = sorted(os.listdir(folder))
        for i in range(0, len(files), 32):             # 32 images at a time
            images = [Image.open(os.path.join(folder, f)) for f in files[i:i + 32]]
            X.append(images_to_features(cnn, images))
            y += [label] * len(images)
        print(f"  {split}/{class_name}: {len(files)} images")
    return np.concatenate(X), np.array(y)


def show_results(name, model, X_test, y_test):
    pred = model.predict(X_test)
    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}   "
          f"Macro F1: {f1_score(y_test, pred, average='macro'):.3f}")
    print(classification_report(y_test, pred, target_names=CLASSES, digits=3))
    print("Confusion matrix (rows = true class, columns = predicted class):")
    print(confusion_matrix(y_test, pred))
    return f1_score(y_test, pred, average="macro")


if __name__ == "__main__":
    start = time.time()
    print("Loading CNN...")
    cnn = load_cnn()

    print("Turning images into features (a few minutes)...")
    X_train, y_train = load_split(cnn, "train")
    X_test, y_test = load_split(cnn, "test")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # -----------------------------------------------------------
    # PART 3: Experiment A - NO feature selection (all 2048 features)
    # -----------------------------------------------------------
    model_a = Pipeline([
        ("scale", StandardScaler()),       # put all features on the same scale
        ("svm", SVC(kernel="rbf", C=1)),
    ])
    model_a.fit(X_train, y_train)
    f1_a = show_results("A) No feature selection (2048 features)", model_a, X_test, y_test)

    # -----------------------------------------------------------
    # PART 4: Experiment B - WITH feature selection (ANOVA)
    # -----------------------------------------------------------
    model_b = Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif)),   # ANOVA: keeps the k best features
        ("svm", SVC(kernel="rbf", probability=True)),
    ])
    # Try different numbers of features and SVM settings (5-fold cross-validation on TRAIN)
    search = GridSearchCV(
        model_b,
        param_grid={"select__k": [32, 64, 128, 256, 512], "svm__C": [1, 10]},
        cv=5, scoring="f1_macro",
    )
    search.fit(X_train, y_train)
    best_k = search.best_params_["select__k"]
    print(f"\nBest setting found on train data: k = {best_k} features, "
          f"C = {search.best_params_['svm__C']}")
    f1_b = show_results(f"B) Feature selection (ANOVA, {best_k} features)",
                        search.best_estimator_, X_test, y_test)

    # -----------------------------------------------------------
    # Summary + save the model with feature selection
    # -----------------------------------------------------------
    print("\n===== SUMMARY =====")
    print(f"A) 2048 features      -> Macro F1 = {f1_a:.3f}")
    print(f"B) {best_k:4d} features (ANOVA) -> Macro F1 = {f1_b:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(search.best_estimator_, os.path.join("models", "caries_model.joblib"))
    print(f"\nModel saved to models/caries_model.joblib  ({time.time() - start:.0f}s)")