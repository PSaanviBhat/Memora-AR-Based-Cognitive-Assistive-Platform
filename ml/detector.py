import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_and_align(self, image_bgr):
        faces = self.app.get(image_bgr)
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        return face.normed_embedding, face
