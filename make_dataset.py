import json
import os
import random

from PIL import Image
from ultralytics import YOLO

random.seed(42)                            # same random choices every run
N_PER_CLASS = {"train": 400, "test": 60}   # how many teeth per class
OUT = "small_data"
CLASSES = ["healthy", "caries", "deep_caries"]

yolo = YOLO(os.path.join("models", "best.pt"))   # tooth detector


def overlap(a, b):
    """How much two boxes overlap: 0 = not at all, 1 = the same box.
    Boxes are [x1, y1, x2, y2]."""
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def cut(image, box):
    """Cuts one tooth out of the X-ray, with a 10% margin around it."""
    x1, y1, x2, y2 = box
    mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    return image.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                       min(image.width, int(x2 + mx)), min(image.height, int(y2 + my))))


for split in ("train", "test"):
    with open(os.path.join("data", f"{split}.json"), encoding="utf-8") as f:
        xrays = json.load(f)
    random.shuffle(xrays)

    need = N_PER_CLASS[split]
    count = {c: 0 for c in CLASSES}
    for c in CLASSES:
        os.makedirs(os.path.join(OUT, split, c), exist_ok=True)

    for xray in xrays:
        if all(count[c] >= need for c in CLASSES):
            break                                  # we have enough teeth

        image = Image.open(xray["path"]).convert("L")   # "L" = grayscale
        name = os.path.splitext(xray["file_name"])[0]

        # Dentist labels of this X-ray: DENTEX boxes are [x, y, w, h]
        labelled = []
        for tooth in xray["teeth"]:
            x, y, w, h = tooth["bbox"]
            labelled.append(([x, y, x + w, y + h], tooth["label"], tooth["fdi"]))

        # 1) Caries and deep caries teeth
        for box, label, fdi in labelled:
            if label in ("caries", "deep_caries") and count[label] < need:
                cut(image, box).save(os.path.join(OUT, split, label, f"{name}_{fdi}.png"))
                count[label] += 1

        # 2) Healthy teeth: at most 2 per X-ray, so they come from many patients
        if count["healthy"] < need:
            found = yolo.predict(xray["path"], imgsz=1024, conf=0.5, verbose=False)[0]
            boxes = found.boxes.xyxy.tolist()
            random.shuffle(boxes)
            taken = 0
            for box in boxes:
                if taken == 2 or count["healthy"] >= need:
                    break
                # healthy = no dentist label (caries, deep caries or other) on this tooth
                if all(overlap(box, lb) < 0.05 for lb, _, _ in labelled):
                    cut(image, box).save(os.path.join(OUT, split, "healthy",
                                                      f"{name}_h{taken}.png"))
                    count["healthy"] += 1
                    taken += 1

    print(split, count)

print("Done! Look inside the 'small_data' folder.")