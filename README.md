# 🦷 Dental Caries Detection on Panoramic X-rays

🎓 Course project, Computer Engineering, Istanbul Arel University.

The system takes a panoramic dental X-ray, finds and numbers every tooth, and marks each tooth as **healthy** ✅, **caries** 🟧 or **deep caries** 🟥:

```
31 teeth found.
Caries (9 teeth): 16, 17, 26, 27, 34, 37, 45, 46, 47
Deep caries (4 teeth): 18, 28, 38, 48
```

> ⚠️ Research project - not a medical diagnosis tool.

## ⚙️ How it works

```
Panoramic X-ray
   -> YOLO11m finds and numbers the teeth (FDI numbers 11-48)
   -> each tooth is cut out (10% margin)
   -> 🤖 ResNet-50 (pretrained on ImageNet)  -> 2048 deep features
      🩺 radiological features (texture, dark areas, brightness) -> 50 handcrafted features
   -> ANOVA feature selection keeps the best 512 of the 2098 features
   -> SVM classifies the tooth: healthy / caries / deep caries
   -> carious teeth are drawn on the X-ray
```

## 💡 Novel method: deep + radiological feature fusion

A pretrained CNN has never seen an X-ray. Dentists recognise caries as **dark areas with a changed texture** inside the tooth. For every tooth we therefore combine:

- 🤖 **2048 deep features** from ResNet-50, and
- 🩺 **50 handcrafted radiological features**: intensity statistics, dark-area ratios, histogram, entropy, GLCM texture (contrast, homogeneity, energy, correlation...) and LBP local patterns.

ANOVA feature selection then chooses the best features from both groups, and an SVM makes the decision.

- **Base paper:** Pornprasertsuk-Damrongsri et al., *Scientific Reports* 2025 - YOLO finds the teeth, then a deep network classifies caries stages on each tooth. We keep this two-stage structure but replace the second stage.
- **Idea for the handcrafted features:** Ateş et al., *Healthcare* 2025 - GLCM texture features + feature selection + SVM on whole panoramic images.
- **Our addition:** handcrafted features are computed **for each tooth**, **fused with deep CNN features**, and used for **caries severity** on the public DENTEX dataset.

## 📁 Files

| File | What it does |
|---|---|
| `prepare_data.py` | 🗂️ Merges the DENTEX training and validation data, labels every tooth and splits the X-rays into train / val / test (70 / 15 / 15) |
| `make_dataset.py` | ✂️ Cuts a balanced tooth dataset: 400 + 400 + 400 teeth for training, 60 + 60 + 60 for testing |
| `train_model.py` | 🧠 Baseline: CNN features with and without ANOVA feature selection + SVM |
| `compare_models.py` | ⚖️ Compares our SVM with classic methods (Logistic Regression, KNN, Decision Tree, Random Forest) |
| `novel_method.py` | 💡 **Novel method**: CNN + radiological features + ANOVA + SVM (`models/fusion_model.joblib`) |
| `more_data.py` | ➕ Experiment: training with up to 2400 non-carious teeth |
| `evaluate.py` | 📊 Runs the whole system on the 113 test X-rays and compares it with the dentists' labels |
| `app.py` | 🖥️ Gradio demo with the novel method |
| `prepare_yolo_teeth.py`, `train_tooth_detector.py` | 🦷 Prepare the data and train the YOLO tooth detector (run on Google Colab) |

🔍 The YOLO tooth detector (`models/best.pt`) was trained on Google Colab (GPU) with `prepare_yolo_teeth.py` and `train_tooth_detector.py`: YOLO11m on 548 DENTEX X-rays with numbered teeth (X-rays of our test set left out), image size 1024, no left-right flips (they would swap the tooth numbers). Tooth detection mAP50 = 0.951.

## 🚀 How to run

1. 📦 Install the libraries: `pip install -r requirements.txt`
2. ⬇️ Download the [DENTEX dataset](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) into the `DENTEX/` folder.
3. 🤖 Put the YOLO model at `models/best.pt` (not in the repository because of its size).
4. ▶️ Run in order:
   ```
   python prepare_data.py
   python make_dataset.py
   python train_model.py
   python compare_models.py
   python novel_method.py     # trains models/fusion_model.joblib (used by the demo)
   python more_data.py        # optional experiment, about 20 minutes
   python evaluate.py
   python app.py              # opens the demo at http://127.0.0.1:7860
   ```

## 📈 Results

All tooth results are on the same test set: 180 teeth (60 per class). 2-class = healthy vs. any caries.

**⚖️ Our classifier vs. classic methods** (same CNN features, best setting of each):

| Model | Features | 3-class Macro F1 | 2-class F1 |
|---|---|---|---|
| **SVM (ours)** | 512 (ANOVA) | **0.579** | **0.843** |
| Logistic Regression | 2048 | 0.556 | 0.806 |
| Random Forest | 2048 | 0.533 | 0.817 |
| KNN (k=5) | 512 (ANOVA) | 0.507 | 0.749 |
| Decision Tree | 512 (ANOVA) | 0.495 | 0.773 |

ANOVA feature selection improved SVM, KNN and Decision Tree, but made Logistic Regression worse (0.556 → 0.506) and did not change Random Forest much: its benefit depends on the classifier.

**💡 Novel method** (SVM + ANOVA, settings chosen by 5-fold cross-validation):

| Features | Kept by ANOVA | 3-class Accuracy | 3-class Macro F1 | 2-class F1 |
|---|---|---|---|---|
| CNN only (2048) | 512 | 0.578 | 0.579 | 0.843 |
| Radiological only (50) | 10 | 0.556 | 0.558 | **0.849** |
| **CNN + radiological (2098)** | 512 (485 CNN + 27 radiological) | **0.606** | **0.606** 🏆 | 0.848 |

- ANOVA kept **54%** of the radiological features but only **24%** of the CNN features.
- The **best of all 2098 features is an LBP texture feature**; 4 of the top 10 are LBP features.
- With only 10 radiological features the model is almost as good as with 512 CNN features.

**🩻 Whole system on 113 test X-rays** (novel method, 420 carious teeth labelled by dentists):

| Threshold | Carious teeth found | Severity also correct | False alarms per X-ray |
|---|---|---|---|
| 0.5 | 79% | 50% | 9.7 |
| **0.6 (used in the demo)** | **75%** | **48%** | **8.7** |
| 0.7 | 71% | 46% | 7.8 |
| 0.8 | 66% | 43% | 6.7 |
| 0.9 | 42% | 26% | 3.9 |

**➕ More data:** training with 1200 or 2400 non-carious teeth (impacted and treated teeth included) reduced false alarms, but the model also found fewer caries. At the same share of caries found, false alarms were almost the same as with a higher threshold (e.g. 66% found: 6.5 vs. 6.7 false alarms per X-ray). More varied healthy teeth gave a small real gain (75% found: 8.35 instead of 8.73 false alarms). Better features helped more than more data.

💬 **Limitations:** the system finds most carious teeth but gives many false alarms. Reasons: the CNN was trained on everyday photos (ImageNet) and never fine-tuned on X-rays, the training set is balanced while most real teeth are healthy, and the model was only tested on DENTEX images (X-rays from other devices may give much worse results).

## 📚 References

- I. E. Hamamci et al., *DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays*, arXiv:2305.19112, 2023.
- I. E. Hamamci et al., *Diffusion-Based Hierarchical Multi-label Object Detection to Analyze Panoramic Dental X-Rays*, MICCAI 2023, pp. 389-399.
- S. Pornprasertsuk-Damrongsri et al., *Clinical application of deep learning for enhanced multistage caries detection in panoramic radiographs*, Scientific Reports 15, 33491, 2025.
- G. Ateş et al., *Decision-Support for Restorative Dentistry: Hybrid Optimization Enhances Detection on Panoramic Radiographs*, Healthcare 13(22), 2904, 2025.
- D. Ozdemir et al., *An enhanced deep learning model for detection and classification of dental caries in panoramic radiographs*, Neural Computing and Applications 38, 3, 2026.
- K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.

## 📜 Licenses

| Part | License |
|---|---|
| DENTEX dataset | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) - non-commercial use, results shared under the same license |
| Ultralytics YOLO (tooth detector) | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) - this project is open source to meet its terms |
| ResNet-50 ImageNet weights | Provided by [torchvision](https://pytorch.org/vision/stable/models.html), used for research and education |

🎓 This is a non-commercial student project.