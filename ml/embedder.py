import numpy as np

def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)
