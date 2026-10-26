import os

import gradio as gr
import joblib
import numpy as np
from PIL import ImageDraw, ImageFont
from ultralytics import YOLO

from novel_method import radiological_features
from train_model import images_to_features, load_cnn

FDI = [q * 10 + t for q in range(1, 5) for t in range(1, 9)]   # YOLO class id -> FDI tooth number
NAMES_TR = {"healthy": "Sağlıklı", "caries": "Çürük", "deep_caries": "Derin çürük"}
COLORS = {"caries": (255, 165, 0), "deep_caries": (230, 30, 30)}   # orange, red
THRESHOLD = 0.6   # a tooth is called carious if P(caries) + P(deep caries) >= 0.6

print("Loading models...")
yolo = YOLO(os.path.join("models", "best.pt"))
cnn = load_cnn()
model = joblib.load(os.path.join("models", "fusion_model.joblib"))


def cut(image, box):
    x1, y1, x2, y2 = box
    mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    return image.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                       min(image.width, int(x2 + mx)), min(image.height, int(y2 + my))))


def tooth_features(tooth_images):
    cnn_part = images_to_features(cnn, tooth_images)
    rad_part = np.array([list(radiological_features(img).values()) for img in tooth_images])
    return np.hstack([cnn_part, rad_part])


def classify(tooth_images):
    probs = model.predict_proba(tooth_features(tooth_images))
    labels = []
    for p_healthy, p_caries, p_deep in probs:
        if p_caries + p_deep >= THRESHOLD:
            labels.append("deep_caries" if p_deep > p_caries else "caries")
        else:
            labels.append("healthy")
    return labels


def find_teeth(image):
    result = yolo.predict(image, imgsz=1024, conf=0.25, agnostic_nms=True, verbose=False)[0]
    best = {}   # fdi -> (confidence, box)
    for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                              result.boxes.conf.tolist()):
        fdi = FDI[int(cls)]
        if fdi not in best or conf > best[fdi][0]:
            best[fdi] = (conf, box)
    return [(fdi, box) for fdi, (conf, box) in sorted(best.items())]


def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def analyze(image):
    if image is None:
        return None, "Lütfen bir röntgen yükleyin."
    image = image.convert("RGB")
    teeth = find_teeth(image)
    note = "\n\nNot: Bu bir araştırma projesidir, tıbbi teşhis yerine geçmez."

    if not teeth:
        label = classify([image])[0]
        return image, f"Tek diş resmi olarak değerlendirildi.\nSonuç: {NAMES_TR[label]}" + note

    labels = classify([cut(image, box) for fdi, box in teeth])

    draw = ImageDraw.Draw(image)
    font = get_font(max(20, image.width // 60))
    used = []   # label rectangles already drawn (to avoid overlapping text)
    for (fdi, box), label in zip(teeth, labels):
        if label == "healthy":
            continue
        color = COLORS[label]
        draw.rectangle(box, outline=color, width=max(3, image.width // 600))
        text = f"{fdi}: {NAMES_TR[label]}"
        x, y = box[0], max(0, box[1] - font.size - 6)
        rect = draw.textbbox((x, y), text, font=font)
        while any(rect[0] < r[2] and rect[2] > r[0] and rect[1] < r[3] and rect[3] > r[1] for r in used):
            y += font.size + 4   # move the label down until it is free
            rect = draw.textbbox((x, y), text, font=font)
        used.append(rect)
        draw.rectangle(rect, fill=color)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

    caries = [fdi for (fdi, box), label in zip(teeth, labels) if label == "caries"]
    deep = [fdi for (fdi, box), label in zip(teeth, labels) if label == "deep_caries"]
    lines = [f"{len(teeth)} diş bulundu."]
    if caries:
        lines.append(f"Çürük ({len(caries)} diş): " + ", ".join(map(str, caries)))
    if deep:
        lines.append(f"Derin çürük ({len(deep)} diş): " + ", ".join(map(str, deep)))
    if not caries and not deep:
        lines.append("Çürük tespit edilmedi.")
    return image, "\n".join(lines) + note


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil", label="Panoramik röntgen (veya tek diş resmi)"),
    outputs=[gr.Image(type="pil", label="Sonuç"), gr.Textbox(label="Bulgular", lines=6)],
    title="Panoramik Röntgende Diş Çürüğü Tespiti",
    description="Turuncu kutu: çürük, kırmızı kutu: derin çürük. "
                "Model: CNN (ResNet-50) + radyolojik özellikler + ANOVA feature selection + SVM (novel method).",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()