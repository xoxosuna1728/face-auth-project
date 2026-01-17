import os
import cv2
import numpy as np
from face_verify import FaceVerifier


def build_reference(enroll_dir="data/enroll/MonaLisa", out_path="models/monalisa_ref.npy"):
    fv = FaceVerifier()
    embeddings = []

    for f in os.listdir(enroll_dir):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(enroll_dir, f)
        img = cv2.imread(path)
        if img is None:
            continue

        face, _ = fv.extract_face(img)
        if face is None:
            continue

        embeddings.append(fv.embed(face))

    if len(embeddings) < 5:
        raise ValueError("Need at least 5 clear MonaLisa images where a face is detected.")

    ref = np.mean(np.array(embeddings), axis=0)
    ref = ref / np.linalg.norm(ref)

    os.makedirs("models", exist_ok=True)
    np.save(out_path, ref)

    print(f"✅ Saved reference to: {out_path}")
    print(f"✅ Used {len(embeddings)} enrollment images")


if __name__ == "__main__":
    build_reference()
