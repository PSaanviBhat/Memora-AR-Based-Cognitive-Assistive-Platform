import os
import numpy as np
from embedder import normalize_embedding

EMBEDDING_DIR = "data/processed/embeddings"
THRESHOLD = 0.45

class FaceRecognizer:
    def __init__(self):
        self.database = {}
        for file in os.listdir(EMBEDDING_DIR):
            name = file.replace(".npy", "")
            self.database[name] = np.load(os.path.join(EMBEDDING_DIR, file))

    def recognize(self, embedding):
        embedding = normalize_embedding(embedding)
        best_match = None
        best_score = -1

        for name, ref_emb in self.database.items():
            score = np.dot(embedding, ref_emb)
            if score > best_score:
                best_score = score
                best_match = name

        if best_score >= THRESHOLD:
            return best_match, best_score
        else:
            return "Unknown", best_score
