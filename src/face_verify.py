import cv2
import numpy as np
from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch


class FaceVerifier:
    def __init__(self, device=None):
        self.detector = MTCNN()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def extract_face(self, bgr_img, required_size=(160, 160)):
        h, w = bgr_img.shape[:2]
        if h < 80 or w < 80:
            return None, None

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        dets = self.detector.detect_faces(rgb)
        if not dets:
            return None, None

        best = max(dets, key=lambda d: d["box"][2] * d["box"][3])
        x, y, w, h = best["box"]
        x, y = max(0, x), max(0, y)

        face_rgb = rgb[y:y+h, x:x+w]
        if face_rgb.size == 0:
            return None, None

        face_rgb = cv2.resize(face_rgb, required_size)
        return face_rgb, (x, y, w, h)

    @torch.no_grad()
    def embed(self, face_rgb_160):
        face = face_rgb_160.astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))
        face = torch.tensor(face).unsqueeze(0).to(self.device)

        emb = self.embedder(face).cpu().numpy()[0]
        emb = emb / np.linalg.norm(emb)
        return emb

    def distance(self, emb1, emb2):
        return float(np.linalg.norm(emb1 - emb2))
