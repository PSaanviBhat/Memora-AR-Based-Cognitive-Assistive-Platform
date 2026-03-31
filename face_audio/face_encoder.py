"""
MEMORA - Face Encoder Module
ArcFace R50 Pipeline for 128-d Face Embeddings
Input: Webcam frame (BGR) | Output: 128-d normalized vector
Latency target: <50ms per frame
"""

import cv2
import numpy as np
import torch
from typing import Optional, List, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class FaceEncoder:
    """
    Isolated face recognition pipeline using ArcFace R50
    - Single-threaded, stateless
    - Deterministic output
    - <50ms per frame latency target
    """
    
    def __init__(self, model_name: str = "buffalo_s", device: str = "cuda"):
        """
        Initialize ArcFace face encoder
        
        Args:
            model_name: InsightFace model ("buffalo_s" recommended)
            device: "cuda" or "cpu"
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
        print(f"[FaceEncoder] ✓ Loaded {model_name} on {device}")
    
    def _load_model(self):
        """Load ArcFace model from InsightFace"""
        try:
            from insightface.app import FaceAnalysis
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if self.device == "cuda" else ['CPUExecutionProvider']
            
            self.model = FaceAnalysis(name=self.model_name, providers=providers)
            self.model.prepare(
                ctx_id=0 if self.device == "cuda" else -1,
                det_size=(320, 320)
            )
        except ImportError:
            print("[FaceEncoder] ✗ InsightFace not installed. Install: pip install insightface")
            raise
        except Exception as e:
            print(f"[FaceEncoder] ✗ Failed to load model: {e}")
            raise
    
    def encode_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from single frame
        
        Args:
            frame: BGR image array (H×W×3)
        
        Returns:
            embedding: 128-d normalized numpy array, or None if no face
        """
        try:
            if frame is None or frame.size == 0:
                return None
            
            faces = self.model.get(frame)
            
            if len(faces) == 0:
                return None
            
            # Select largest face (most prominent)
            face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
            
            # Normalize embedding to unit vector
            embedding = face.embedding / np.linalg.norm(face.embedding)
            
            return embedding.astype(np.float32)  # 128-d vector
            
        except Exception as e:
            print(f"[FaceEncoder] ✗ Frame encoding failed: {e}")
            return None
    
    def encode_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process multiple frames (for registration)
        
        Args:
            frames: List of BGR images
        
        Returns:
            List of 128-d embeddings (None filtered out)
        """
        embeddings = []
        for frame in frames:
            emb = self.encode_frame(frame)
            if emb is not None:
                embeddings.append(emb)
        return embeddings


# Unit test
if __name__ == "__main__":
    print("[FaceEncoder] Testing with synthetic data (random noise)...")
    encoder = FaceEncoder(device="cpu")
    
    # Test 1: Synthetic data (should return None - expected)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    emb = encoder.encode_frame(test_frame)
    
    if emb is None:
        print("✓ Synthetic test: Correctly returned None (no face in random data)")
    else:
        print(f"✗ Synthetic test: Unexpected embedding shape {emb.shape}")
    
    # Test 2: Test with real webcam if available
    print("\n[FaceEncoder] Testing with real webcam (press 'q' to stop)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Webcam not available")
    else:
        frame_count = 0
        faces_detected = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            emb = encoder.encode_frame(frame)
            frame_count += 1
            
            if emb is not None:
                faces_detected += 1
                status = f"FACE DETECTED | {faces_detected}/{frame_count} frames"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                print(f"  Frame {frame_count}: ✓ Shape {emb.shape}, Norm {np.linalg.norm(emb):.4f}")
            else:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
            
            cv2.imshow('FaceEncoder Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Detected faces in {faces_detected}/{frame_count} frames")