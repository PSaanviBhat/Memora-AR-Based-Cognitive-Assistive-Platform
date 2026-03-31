"""
MEMORA - Identity Confidence Module
IC(u,I) Fusion Function: Weighted cosine similarity of face + speaker embeddings
Output: JSON contract for T3 (Trust Engine) handoff
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass, asdict
import json


@dataclass
class IdentityConfidence:
    """
    Structured output from IC fusion
    Required fields for T3 (Trust Engine) integration
    """
    user_id: str                # Unique identifier
    IC_score: float             # 0-1: main fused confidence score
    face_confidence: float      # 0-1: face similarity
    voice_confidence: float     # 0-1: voice similarity
    is_known: bool              # Known user (above theta_known threshold)
    confidence_level: str       # "HIGH" / "MEDIUM" / "LOW"


def to_json(ic: IdentityConfidence) -> dict:
    """
    Convert IdentityConfidence to JSON for T3 handoff
    
    Args:
        ic: IdentityConfidence object
    
    Returns:
        dict with keys: user_id, IC_score, face_confidence, voice_confidence, is_known, confidence_level
    """
    return {
        "user_id": ic.user_id,
        "IC_score": round(ic.IC_score, 4),
        "face_confidence": round(ic.face_confidence, 4),
        "voice_confidence": round(ic.voice_confidence, 4),
        "is_known": ic.is_known,
        "confidence_level": ic.confidence_level
    }


class IdentityConfusionFuser:
    """
    Fuses face (128-d) + voice (192-d) embeddings into single IC score
    Formula: IC(u,I) = w * sim(face) + (1-w) * sim(voice)
    where w = fusion_weight (default 0.6 = 60% face, 40% voice)
    """

    def __init__(self, fusion_weight: float = 0.6,
                 theta_known: float = 0.70,
                 theta_high: float = 0.85,
                 theta_medium: float = 0.70):
        """
        Initialize IC fuser with configurable thresholds

        Args:
            fusion_weight: Weight for face embedding (0-1)
                Default 0.6 = 60% face, 40% voice
            theta_known: Threshold for "known" user classification (default 0.70)
            theta_high: Threshold for HIGH confidence (default 0.85)
            theta_medium: Threshold for MEDIUM confidence (default 0.70)
        """
        assert 0.0 <= fusion_weight <= 1.0, "fusion_weight must be in [0, 1]"
        assert 0.0 <= theta_known <= 1.0, "theta_known must be in [0, 1]"
        assert 0.0 <= theta_high <= 1.0, "theta_high must be in [0, 1]"
        assert 0.0 <= theta_medium <= 1.0, "theta_medium must be in [0, 1]"
        assert theta_high >= theta_medium >= theta_known, \
            "Expected: theta_high >= theta_medium >= theta_known"

        self.fusion_weight = fusion_weight
        self.theta_known = theta_known
        self.theta_high = theta_high
        self.theta_medium = theta_medium
    
    def fuse(self, face_emb: np.ndarray, voice_emb: np.ndarray,
             face_ref_emb: np.ndarray = None, voice_ref_emb: np.ndarray = None) -> float:
        """
        Compute fused similarity score from face and voice embeddings
        
        Args:
            face_emb: 128-d normalized probe face vector
            voice_emb: 192-d normalized probe voice vector
            face_ref_emb: Optional reference face vector
            voice_ref_emb: Optional reference voice vector
        
        Returns:
            IC_score: float in [0, 1]
        """
        try:
            # If reference vectors are not provided, preserve prior behavior.
            if face_ref_emb is None:
                face_ref_emb = face_emb
            if voice_ref_emb is None:
                voice_ref_emb = voice_emb

            # Cosine similarity = dot product for unit-normalized embeddings.
            face_sim = float(np.clip(np.dot(face_emb, face_ref_emb), 0.0, 1.0))
            voice_sim = float(np.clip(np.dot(voice_emb, voice_ref_emb), 0.0, 1.0))
            
            # Weighted fusion
            fused = self.fusion_weight * face_sim + (1 - self.fusion_weight) * voice_sim
            
            return float(np.clip(fused, 0.0, 1.0))
        
        except Exception as e:
            print(f"[IdentityConfidence] ✗ Fusion failed: {e}")
            return 0.0
    
    def compute_ic(self, user_id: str, face_emb: np.ndarray, voice_emb: np.ndarray,
                   face_ref_emb: np.ndarray = None, voice_ref_emb: np.ndarray = None,
                   theta_known: float = 0.70, theta_high: float = 0.85, 
                   theta_medium: float = 0.70) -> IdentityConfidence:
        """
        Compute full Identity Confidence object with confidence level
        
        Args:
            user_id: User identifier string
            face_emb: 128-d probe face embedding
            voice_emb: 192-d probe voice embedding
            face_ref_emb: Optional reference face embedding
            voice_ref_emb: Optional reference voice embedding
            theta_known: Threshold for "known" user classification (default 0.70)
            theta_high: Threshold for HIGH confidence (default 0.85)
            theta_medium: Threshold for MEDIUM confidence (default 0.70)
        
        Returns:
            IdentityConfidence object with all fields populated
        """
        # Compute fused score
        if face_ref_emb is None:
            face_ref_emb = face_emb
        if voice_ref_emb is None:
            voice_ref_emb = voice_emb

        ic_score = self.fuse(face_emb, voice_emb, face_ref_emb, voice_ref_emb)
        
        # Extract individual confidences (for embeddings already normalized)
        face_conf = float(np.clip(np.dot(face_emb, face_ref_emb), 0.0, 1.0))
        voice_conf = float(np.clip(np.dot(voice_emb, voice_ref_emb), 0.0, 1.0))
        
        # Determine confidence level based on thresholds
        if ic_score >= theta_high:
            conf_level = "HIGH"
        elif ic_score >= theta_medium:
            conf_level = "MEDIUM"
        else:
            conf_level = "LOW"
        
        # Determine if user is "known"
        is_known = ic_score >= theta_known
        
        return IdentityConfidence(
            user_id=user_id,
            IC_score=ic_score,
            face_confidence=face_conf,
            voice_confidence=voice_conf,
            is_known=is_known,
            confidence_level=conf_level
        )

    def log_confidence(self, user_id: str, face_emb: np.ndarray, voice_emb: np.ndarray,
                      face_ref_emb: np.ndarray = None, voice_ref_emb: np.ndarray = None,
                      face_quality: float = 1.0, voice_quality: float = 1.0) -> Dict:
        """
        Compute IC and return detailed structured log for evaluation

        Args:
            user_id: User identifier
            face_emb: 128-d probe face embedding
            voice_emb: 192-d probe voice embedding
            face_ref_emb: Optional reference face embedding
            voice_ref_emb: Optional reference voice embedding
            face_quality: Optional quality score [0, 1] for face
            voice_quality: Optional quality score [0, 1] for voice

        Returns:
            Dict with detailed IC computation (for evaluation/logging)
        """
        from datetime import datetime

        # Compute standard IC
        ic = self.compute_ic(user_id, face_emb, voice_emb,
                            face_ref_emb, voice_ref_emb,
                            self.theta_known, self.theta_high, self.theta_medium)

        # Return as dict with extra quality annotations
        return {
            'user_id': user_id,
            'IC_score': round(ic.IC_score, 4),
            'face_sim': round(ic.face_confidence, 4),
            'voice_sim': round(ic.voice_confidence, 4),
            'face_quality': round(face_quality, 4),
            'voice_quality': round(voice_quality, 4),
            'is_known': ic.is_known,
            'confidence_level': ic.confidence_level,
            'theta_applied': {
                'theta_known': self.theta_known,
                'theta_high': self.theta_high,
                'theta_medium': self.theta_medium
            },
            'timestamp': datetime.now().isoformat()
        }


# Unit test
if __name__ == "__main__":
    print("[IdentityConfidence] Testing IC fusion...")
    fuser = IdentityConfusionFuser(fusion_weight=0.6)
    
    # Test 1: Perfect embeddings (all 1s, normalized)
    face_emb = np.ones(128, dtype=np.float32) / np.sqrt(128)
    voice_emb = np.ones(192, dtype=np.float32) / np.sqrt(192)
    
    ic = fuser.compute_ic("test_user", face_emb, voice_emb)
    
    print(f"✓ Test 1 - Perfect embeddings:")
    print(f"  IC_score: {ic.IC_score:.4f}")
    print(f"  Confidence level: {ic.confidence_level}")
    print(f"  Is known: {ic.is_known}")
    
    # Test 2: Random embeddings
    face_emb_random = np.random.randn(128).astype(np.float32)
    face_emb_random /= np.linalg.norm(face_emb_random)
    voice_emb_random = np.random.randn(192).astype(np.float32)
    voice_emb_random /= np.linalg.norm(voice_emb_random)
    
    ic2 = fuser.compute_ic("random_user", face_emb_random, voice_emb_random)
    
    print(f"\n✓ Test 2 - Random embeddings:")
    print(f"  IC_score: {ic2.IC_score:.4f}")
    print(f"  Confidence level: {ic2.confidence_level}")
    
    # Test 3: JSON output
    json_out = to_json(ic)
    print(f"\n✓ Test 3 - JSON output:")
    print(json.dumps(json_out, indent=2))
    
    # Assertions
    assert 0.0 <= ic.IC_score <= 1.0, "IC score out of range"
    assert ic.confidence_level in ["HIGH", "MEDIUM", "LOW"], "Invalid confidence level"
    assert isinstance(json_out, dict), "JSON output should be dict"
    assert "IC_score" in json_out, "Missing IC_score in JSON"
    print("\n✓ All assertions passed")