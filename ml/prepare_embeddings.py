import os
import cv2
import numpy as np
from detector import FaceDetector
from embedder import normalize_embedding

DATA_DIR = "data/raw/family"
OUT_DIR = "data/processed/embeddings"

os.makedirs(OUT_DIR, exist_ok=True)

detector = FaceDetector()

for person in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        result = detector.detect_and_align(img)
        if result is None:
            continue

        embedding, _ = result
        embedding = normalize_embedding(embedding)
        embeddings.append(embedding)

    if len(embeddings) == 0:
        continue

    embeddings = np.vstack(embeddings)
    prototype = np.mean(embeddings, axis=0)
    prototype = normalize_embedding(prototype)

    np.save(os.path.join(OUT_DIR, f"{person}.npy"), prototype)
    print(f"[OK] Saved prototype for {person}")
