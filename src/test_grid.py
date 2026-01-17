import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from face_verify import FaceVerifier


def draw_box(bgr, box):
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)


def match_score_from_distance(dist, threshold):
    # simple confidence score: closer = higher confidence
    # 1.0 means perfect match, 0.0 means very far
    return max(0.0, 1.0 - (dist / threshold))


def collect_images(test_root):
    """
    Expected structure:
      data/test/MonaLisa/*.jpg
      data/test/Unknown/*.jpg
    """
    items = []
    for label in ["MonaLisa", "Unknown"]:
        folder = os.path.join(test_root, label)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((label, os.path.join(folder, f)))
    return items


if __name__ == "__main__":
    # Load verifier + MonaLisa reference embedding
    fv = FaceVerifier()
    ref_path = "models/monalisa_ref.npy"
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            "models/monalisa_ref.npy not found. Run: python src/build_reference.py"
        )
    ref = np.load(ref_path)

    # Threshold for deciding MonaLisa vs Unknown
    # If dist < threshold => MonaLisa
    threshold = 0.90  # tune later if needed

    test_root = "data/test"
    items = collect_images(test_root)

    if len(items) == 0:
        raise ValueError(
            "No test images found. Put images into:\n"
            "  data/test/MonaLisa/\n"
            "  data/test/Unknown/\n"
        )

    # Show only first 5 like your screenshot
    show_n = 5
    items = items[:show_n]

    print("Initializing demo system...")
    print("Getting test images...")
    print("Making predictions...")
    print("Displaying results...")

    imgs = []
    titles = []
    details = []

    correct = 0
    total = 0

    for true_label, path in items:
        bgr = cv2.imread(path)
        if bgr is None:
            continue

        face, box = fv.extract_face(bgr)

        if face is None:
            pred = "NoFace"
            conf = 0.0
        else:
            emb = fv.embed(face)
            dist = fv.distance(emb, ref)

            pred = "MonaLisa" if dist < threshold else "Unknown"
            conf = match_score_from_distance(dist, threshold)

        total += 1
        if pred == true_label:
            correct += 1

        if box is not None:
            draw_box(bgr, box)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs.append(rgb)

        titles.append(
            f"True: {true_label}\nPred: {pred}\nConf: {conf*100:.2f}%"
        )

        status = "✓" if pred == true_label else "✗"
        details.append((status, true_label, pred, conf))

    acc = correct / total if total else 0.0

    print(f"\nTest Accuracy: {acc*100:.2f}%\n")
    print("Detailed Results:")
    for status, t, p, c in details:
        print(f"{status} True: {t:<10} Predicted: {p:<10} Confidence: {c*100:.2f}%")

    fig, axes = plt.subplots(1, len(imgs), figsize=(18, 4))
    if len(imgs) == 1:
        axes = [axes]

    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
