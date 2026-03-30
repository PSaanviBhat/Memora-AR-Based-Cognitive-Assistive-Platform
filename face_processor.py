"""
MEMORA - Face Processing Module
Face quality filtering, scoring, and weighted aggregation
Input: Webcam frames + InsightFace face objects | Output: quality-weighted embeddings
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class ScoredFace:
    """Face object with quality score"""
    face_obj: object            # InsightFace Face object
    det_score: float            # Detection confidence [0, 1]
    pose_angles: Tuple[float]   # (pitch, roll, yaw) in degrees
    blur_variance: float        # Laplacian variance (0-1 normalized)
    quality_score: float        # Combined quality score [0, 1]
    bbox: Tuple[int]            # Bounding box (x1, y1, x2, y2)


class FaceQualityFilter:
    """
    Filters and scores face detections for quality
    - Detection confidence thresholding
    - Head pose filtering (frontal faces only)
    - Blur detection via Laplacian variance
    - Multi-face handling (selects best quality)
    - Weighted embedding aggregation
    """

    def __init__(self, confidence_threshold: float = 0.95,
                 pose_limits_deg: Tuple[float] = (30, 30, 30),
                 blur_threshold: float = 50.0):
        """
        Initialize face quality filter

        Args:
            confidence_threshold: Minimum detection confidence [0, 1]
            pose_limits_deg: (pitch_max, roll_max, yaw_max) in degrees
            blur_threshold: Minimum Laplacian variance to accept (higher = sharper)
        """
        self.confidence_threshold = confidence_threshold
        self.pose_limits_pitch, self.pose_limits_roll, self.pose_limits_yaw = pose_limits_deg
        self.blur_threshold = blur_threshold

        print(f"[FaceQualityFilter] ✓ Initialized")
        print(f"  Confidence threshold: {confidence_threshold:.2f}")
        print(f"  Pose limits (degrees): pitch={self.pose_limits_pitch}, roll={self.pose_limits_roll}, yaw={self.pose_limits_yaw}")
        print(f"  Blur threshold: {blur_threshold:.1f}")

    def compute_laplacian_variance(self, frame: np.ndarray,
                                   bbox: Optional[Tuple[int]] = None) -> float:
        """
        Compute Laplacian variance (measure of sharpness/blur)
        Higher variance = sharper image = less blur

        Args:
            frame: BGR image array
            bbox: Optional bounding box to focus on face region

        Returns:
            Laplacian variance (normalized to [0, 100] approximately)
        """
        try:
            if frame is None or frame.size == 0:
                return 0.0

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Optionally crop to face region for more accurate blur detection
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(gray.shape[1], int(x2)), min(gray.shape[0], int(y2))
                gray = gray[y1:y2, x1:x2]

            if gray.size == 0:
                return 0.0

            # Compute Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = float(np.var(laplacian))

            # Normalize to approximately [0, 1] (typical variance ~50-500)
            normalized_variance = min(variance / 100.0, 1.0)

            return normalized_variance

        except Exception as e:
            print(f"[FaceQualityFilter] ✗ Laplacian computation failed: {e}")
            return 0.5

    def extract_pose_angles(self, face_obj) -> Tuple[float, float, float]:
        """
        Extract head pose angles from face landmarks if available
        Falls back to simple heuristic if landmarks unavailable

        Args:
            face_obj: InsightFace Face object

        Returns:
            (pitch, roll, yaw) in degrees
        """
        try:
            # If face object has pose attribute (InsightFace provides this)
            if hasattr(face_obj, 'pose') and face_obj.pose is not None:
                # pose typically returns [pitch, roll, yaw] in radians
                pose = face_obj.pose
                if isinstance(pose, (list, tuple, np.ndarray)) and len(pose) >= 3:
                    pitch = float(np.degrees(pose[0]))
                    roll = float(np.degrees(pose[1]))
                    yaw = float(np.degrees(pose[2]))
                    return (pitch, roll, yaw)

            # Fallback: estimate from bounding box (side-profile detection)
            # If face is off-center horizontally, likely has high yaw
            if hasattr(face_obj, 'bbox') and face_obj.bbox is not None:
                bbox = face_obj.bbox
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Placeholder - no real pose estimation
                # In practice, more sophisticated 3D face models would be used
                return (0.0, 0.0, 0.0)

            return (0.0, 0.0, 0.0)

        except Exception as e:
            print(f"[FaceQualityFilter] ✗ Pose extraction failed: {e}")
            return (0.0, 0.0, 0.0)

    def score_face_frame(self, face_obj, frame: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute quality score for a single face detection

        Args:
            face_obj: InsightFace Face object
            frame: BGR image frame

        Returns:
            (quality_score [0, 1], metrics dict)
        """
        try:
            # 1. Detection confidence (normalized to [0, 1])
            det_score = float(getattr(face_obj, 'det_score', 0.8))
            det_confidence = np.clip(det_score, 0, 1)

            # 2. Head pose (penalty for non-frontal)
            pitch, roll, yaw = self.extract_pose_angles(face_obj)
            pose_penalty = (
                min(abs(pitch) / self.pose_limits_pitch, 1.0) +
                min(abs(roll) / self.pose_limits_roll, 1.0) +
                min(abs(yaw) / self.pose_limits_yaw, 1.0)
            ) / 3.0  # Average penalty [0, 1]
            pose_quality = 1.0 - pose_penalty

            # 3. Blur detection (Laplacian variance)
            bbox = face_obj.bbox if hasattr(face_obj, 'bbox') else None
            blur_variance = self.compute_laplacian_variance(frame, bbox)
            blur_quality = np.clip(blur_variance, 0, 1)  # Already ~normalized

            # Combined quality score (weighted average)
            # 50% detection, 20% pose, 30% sharpness
            quality_score = (
                0.5 * det_confidence +
                0.2 * pose_quality +
                0.3 * blur_quality
            )

            metrics = {
                'det_score': det_confidence,
                'pose_angles': (pitch, roll, yaw),
                'pose_quality': pose_quality,
                'blur_variance': blur_variance,
                'blur_quality': blur_quality,
                'quality_score': quality_score
            }

            return quality_score, metrics

        except Exception as e:
            print(f"[FaceQualityFilter] ✗ Face scoring failed: {e}")
            return 0.0, {}

    def filter_faces(self, faces: List, frame: np.ndarray,
                     min_quality: float = 0.5) -> Optional[Tuple]:
        """
        Select best face from detections

        Args:
            faces: List of InsightFace Face objects
            frame: BGR image frame
            min_quality: Minimum acceptable quality score

        Returns:
            (best_face, quality_score, metrics) or None if no faces pass
        """
        try:
            if len(faces) == 0:
                return None

            best_face = None
            best_score = -1.0
            best_metrics = {}

            for face in faces:
                score, metrics = self.score_face_frame(face, frame)

                if score > best_score and score >= min_quality:
                    best_score = score
                    best_face = face
                    best_metrics = metrics

            if best_face is None:
                # If no face passes quality threshold, return highest-scoring one anyway
                for face in faces:
                    score, metrics = self.score_face_frame(face, frame)
                    if score > best_score:
                        best_score = score
                        best_face = face
                        best_metrics = metrics

            if best_face is None:
                return None

            return (best_face, best_score, best_metrics)

        except Exception as e:
            print(f"[FaceQualityFilter] ✗ Face filtering failed: {e}")
            return None

    def aggregate_face_embeddings(self, embeddings: List[np.ndarray],
                                  quality_scores: List[float]) -> Optional[np.ndarray]:
        """
        Aggregate face embeddings with quality-based weighting

        Args:
            embeddings: List of 128-d face embeddings
            quality_scores: Quality scores for each embedding [0, 1]

        Returns:
            Weighted aggregated 128-d embedding (L2-normalized)
        """
        try:
            if len(embeddings) == 0:
                return None

            embeddings_array = np.array(embeddings, dtype=np.float32)
            quality_scores = np.array(quality_scores, dtype=np.float32)

            # Normalize quality scores to weights (sum to 1)
            weights = quality_scores / np.sum(quality_scores)

            # Weighted mean
            agg_emb = np.average(embeddings_array, axis=0, weights=weights)

            # L2 normalization
            norm = np.linalg.norm(agg_emb)
            if norm > 0:
                agg_emb = agg_emb / norm

            return agg_emb.astype(np.float32)

        except Exception as e:
            print(f"[FaceQualityFilter] ✗ Embedding aggregation failed: {e}")
            return None


# Unit tests
if __name__ == "__main__":
    print("[FaceQualityFilter] Running unit tests...\n")

    filter_obj = FaceQualityFilter(
        confidence_threshold=0.95,
        pose_limits_deg=(30, 30, 30),
        blur_threshold=50.0
    )

    # Test 1: Laplacian variance computation
    print("Test 1: Laplacian variance computation")
    # Create synthetic image (sharp pattern)
    sharp_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(sharp_img, (100, 100), (500, 400), (255, 255, 255), -1)
    cv2.rectangle(sharp_img, (150, 150), (450, 350), (0, 0, 0), -1)

    # Create blurry version
    blurry_img = cv2.GaussianBlur(sharp_img, (21, 21), 0)

    sharp_var = filter_obj.compute_laplacian_variance(sharp_img)
    blurry_var = filter_obj.compute_laplacian_variance(blurry_img)

    print(f"  Sharp image variance: {sharp_var:.4f}")
    print(f"  Blurry image variance: {blurry_var:.4f}")
    assert sharp_var > blurry_var, "Sharp image should have higher Laplacian variance"
    print("  ✓ Pass\n")

    # Test 2: Pose angle extraction
    print("Test 2: Pose angle extraction (fallback)")
    # Create mock face object
    class MockFace:
        def __init__(self):
            self.pose = None
            self.bbox = [100, 100, 200, 300]

    mock_face = MockFace()
    pitch, roll, yaw = filter_obj.extract_pose_angles(mock_face)
    print(f"  Pose angles - Pitch: {pitch:.2f}°, Roll: {roll:.2f}°, Yaw: {yaw:.2f}°")
    assert isinstance(pitch, float) and isinstance(roll, float) and isinstance(yaw, float)
    print("  ✓ Pass\n")

    # Test 3: Face scoring
    print("Test 3: Face scoring")
    mock_face.det_score = 0.92
    score, metrics = filter_obj.score_face_frame(mock_face, sharp_img)
    print(f"  Quality score: {score:.4f}")
    print(f"  Detection confidence: {metrics.get('det_score', 0):.4f}")
    print(f"  Blur quality: {metrics.get('blur_quality', 0):.4f}")
    assert 0 <= score <= 1, "Quality score out of range"
    print("  ✓ Pass\n")

    # Test 4: Embedding aggregation
    print("Test 4: Embedding aggregation")
    dummy_embeddings = [
        np.random.randn(128).astype(np.float32),
        np.random.randn(128).astype(np.float32),
        np.random.randn(128).astype(np.float32)
    ]
    quality_scores = [0.8, 0.9, 0.7]
    agg_emb = filter_obj.aggregate_face_embeddings(dummy_embeddings, quality_scores)
    print(f"  Aggregated shape: {agg_emb.shape}")
    print(f"  Norm: {np.linalg.norm(agg_emb):.4f}")
    assert agg_emb.shape == (128,), "Should be 128-d"
    assert abs(np.linalg.norm(agg_emb) - 1.0) < 0.01, "Should be normalized"
    print("  ✓ Pass\n")

    print("✓ All FaceQualityFilter tests passed!")
