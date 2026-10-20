import json
import os
import time

from PIL import Image

from app import cnn, cut, find_teeth, model
from train_model import images_to_features

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
MIN_OVERLAP = 0.3


def overlap(a, b):
    """IoU of two boxes [x1, y1, x2, y2]."""
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def to_corners(bbox):
    """DENTEX box [x, y, width, height] -> [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def decide(probs, threshold):
    """Same rule as classify() in app.py."""
    labels = []
    for p_healthy, p_caries, p_deep in probs:
        if p_caries + p_deep >= threshold:
            labels.append("deep_caries" if p_deep > p_caries else "caries")
        else:
            labels.append("healthy")
    return labels


def compare(teeth, labels, dentist):
    """Returns (found, severity correct, false alarms) for one X-ray."""
    predicted = [(box, label) for (fdi, box), label in zip(teeth, labels) if label != "healthy"]
    used = set()
    found = severity_ok = 0
    for d_box, d_label in dentist:
        best, best_overlap = None, 0.0
        for j, (p_box, p_label) in enumerate(predicted):
            if j not in used and overlap(d_box, p_box) > best_overlap:
                best, best_overlap = j, overlap(d_box, p_box)
        if best is not None and best_overlap >= MIN_OVERLAP:
            used.add(best)
            found += 1
            severity_ok += predicted[best][1] == d_label
    return found, severity_ok, len(predicted) - len(used)


if __name__ == "__main__":
    with open(os.path.join("data", "test.json"), encoding="utf-8") as f:
        images = json.load(f)

    # 1) Run YOLO + CNN + SVM once on every test X-ray and keep the probabilities
    start = time.time()
    xrays = []   # (file name, teeth, probabilities, dentist's carious teeth)
    for n, info in enumerate(images, 1):
        image = Image.open(info["path"]).convert("RGB")
        teeth = find_teeth(image)
        probs = model.predict_proba(images_to_features(cnn, [cut(image, box) for fdi, box in teeth])) \
            if teeth else []
        dentist = [(to_corners(t["bbox"]), t["label"]) for t in info["teeth"]
                   if t["label"] in ("caries", "deep_caries")]
        xrays.append((info["file_name"], teeth, probs, dentist))
        if n % 10 == 0 or n == len(images):
            print(f"  {n}/{len(images)} X-rays  ({time.time() - start:.0f}s)")

    # 2) Compare with the dentists' labels for every threshold
    total = sum(len(dentist) for _, _, _, dentist in xrays)
    lines = [f"Test set: {len(xrays)} X-rays, {total} carious teeth (dentists' labels)", "",
             "Threshold | Carious teeth found | Severity also correct | False alarms per X-ray",
             "----------+---------------------+-----------------------+-----------------------"]
    per_xray = []   # errors of every X-ray at the threshold used in app.py
    for threshold in THRESHOLDS:
        found_all = severity_all = alarms_all = 0
        for name, teeth, probs, dentist in xrays:
            found, severity_ok, alarms = compare(teeth, decide(probs, threshold), dentist)
            found_all += found
            severity_all += severity_ok
            alarms_all += alarms
            if threshold == 0.6:
                per_xray.append((name, len(dentist), found, alarms))
        lines.append(f"   {threshold:.1f}    | {found_all:4d}/{total} ({100 * found_all / total:3.0f}%)  "
                     f"| {severity_all:4d}/{total} ({100 * severity_all / total:3.0f}%)    "
                     f"| {alarms_all / len(xrays):.2f}")

    # 3) Best and worst X-rays at threshold 0.6 (examples for the presentation)
    per_xray.sort(key=lambda r: (r[1] - r[2]) + r[3])   # missed + false alarms
    good = [r for r in per_xray if r[1] >= 3][:3]
    lines += ["", "Threshold 0.6 - best X-rays (at least 3 carious teeth):"]
    lines += [f"  {name}: found {found}/{carious}, false alarms {alarms}" for name, carious, found, alarms in good]
    lines += ["Threshold 0.6 - worst X-rays:"]
    lines += [f"  {name}: found {found}/{carious}, false alarms {alarms}" for name, carious, found, alarms in per_xray[-3:]]

    print("\n" + "\n".join(lines))
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "evaluation.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nSaved to results/evaluation.txt")