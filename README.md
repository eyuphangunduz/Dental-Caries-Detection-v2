# 🦷 Dental Caries Detection on Panoramic X-rays

🎓 Course project, Computer Engineering, Istanbul Arel University.

The system takes a panoramic dental X-ray, finds and numbers every tooth, and marks each tooth as **healthy** ✅, **caries** 🟧 or **deep caries** 🟥:

```
31 teeth found.
Caries (10 teeth): 14, 15, 16, 17, 25, 26, 27, 37, 46, 47
Deep caries (4 teeth): 18, 28, 38, 48
```

> ⚠️ Research project - not a medical diagnosis tool.

## ⚙️ How it works

```
Panoramic X-ray
   -> YOLO11m finds and numbers the teeth (FDI numbers 11-48)
   -> each tooth is cut out (10% margin)
   -> ResNet-50 (pretrained on ImageNet) turns each tooth into 2048 features
   -> ANOVA feature selection keeps the best k features
   -> SVM classifies the tooth: healthy / caries / deep caries
   -> carious teeth are drawn on the X-ray
```

## 📁 Files

| File | What it does |
|---|---|
| `prepare_data.py` | 🗂️ Merges the DENTEX training and validation data, labels every tooth and splits the X-rays into train / val / test (70 / 15 / 15) |
| `make_dataset.py` | ✂️ Cuts a balanced tooth dataset: 400 + 400 + 400 teeth for training, 60 + 60 + 60 for testing |
| `train_model.py` | 🧠 Extracts CNN features and trains two models: SVM with all 2048 features vs. SVM with ANOVA feature selection (k chosen by 5-fold cross-validation) |
| `app.py` | 🖥️ Gradio demo: upload a panoramic X-ray, see the carious teeth |
| `evaluate.py` | 📊 Runs the whole system on the 113 test X-rays and compares it with the dentists' labels |

🔍 The YOLO tooth detector (`models/best.pt`) was trained on Google Colab (GPU) in the first version of this project:
[Dental-Caries-Detection](https://github.com/eyuphangunduz/Dental-Caries-Detection) (`prepare_yolo_teeth.py`, `train_tooth_detector.py`).

## 🚀 How to run

1. 📦 Install the libraries: `pip install -r requirements.txt`
2. ⬇️ Download the [DENTEX dataset](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) into the `DENTEX/` folder.
3. 🤖 Put the YOLO model at `models/best.pt` (not in the repository because of its size).
4. ▶️ Run in order:
```
   python prepare_data.py
   python make_dataset.py
   python train_model.py
   python evaluate.py
   python app.py        # opens the demo at http://127.0.0.1:7860
```

## 📈 Results

**🦷 Tooth classification** (180 test teeth, 60 per class):

| Model | Features | Accuracy | Macro F1 |
|---|---|---|---|
| SVM | all 2048 | 0.567 | 0.567 |
| ANOVA + SVM | best 512 | 0.578 | **0.579** 🏆 |

**🩻 Whole system on 113 test X-rays** (420 carious teeth labelled by dentists):

| Threshold | Carious teeth found | Severity also correct | False alarms per X-ray |
|---|---|---|---|
| 0.5 | 80% | 51% | 10.0 |
| **0.6 (used in the demo)** | **75%** | **48%** | **8.9** |
| 0.7 | 71% | 46% | 7.8 |
| 0.8 | 67% | 44% | 6.6 |
| 0.9 | 39% | 23% | 3.6 |

💡 The system finds most carious teeth but gives many false alarms. Main reasons: the training set is balanced (2/3 carious teeth) while most real teeth are healthy, the CNN was trained on everyday photos (ImageNet) and never fine-tuned on X-rays, and teeth types not seen in training (e.g. impacted wisdom teeth) are often marked as carious.

## 📚 Dataset and licenses

X-rays and labels come from the **DENTEX** dataset (MICCAI 2023 challenge). Please cite:

- I. E. Hamamci et al., *DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays*, arXiv:2305.19112, 2023.
- I. E. Hamamci et al., *Diffusion-Based Hierarchical Multi-label Object Detection to Analyze Panoramic Dental X-Rays*, MICCAI 2023, pp. 389-399.

| Part | License |
|---|---|
| DENTEX dataset | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) - non-commercial use, results shared under the same license |
| Ultralytics YOLO (tooth detector) | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) - this project is open source to meet its terms |
| ResNet-50 ImageNet weights | Provided by [torchvision](https://pytorch.org/vision/stable/models.html), used for research and education |

🎓 This is a non-commercial student project.