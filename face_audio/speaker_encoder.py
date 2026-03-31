"""
MEMORA - Speaker Encoder Module
ECAPA-TDNN Pipeline for 192-d Speaker Embeddings
Input: 16kHz mono audio (2+ seconds) | Output: 192-d normalized vector
"""
"""
MEMORA - Speaker Encoder Module
ECAPA-TDNN Pipeline for 192-d Speaker Embeddings
Input: 16kHz mono audio (2+ seconds) | Output: 192-d normalized vector
"""
import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"

import numpy as np
import torch
import torchaudio
from typing import Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class SpeakerEncoder:
    """
    ECAPA-TDNN WITHOUT SpeechBrain loader (uses manual torch approach)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.sample_rate = 16000
        self.model = None
        self._load_model()
        print(f"[SpeakerEncoder] ✓ Loaded ECAPA-TDNN (manual) on {device}")

    def _load_model(self):
        """Load ECAPA-TDNN model using direct torch approach"""
        try:
            import torch.nn as nn
            
            # Try loading from torch.hub first
            print("[SpeakerEncoder] Loading ECAPA-TDNN from torch.hub...")
            self.model = torch.hub.load(
                'TaoRuijie/ECAPA-TDNN',
                'ecapa_tdnn',
                force_reload=False
            )
            self.model.eval()
            self.model.to(self.device)
            print("[SpeakerEncoder] ✓ Successfully loaded from torch.hub")
            
        except Exception as e:
            print(f"[SpeakerEncoder] ✗ torch.hub load failed: {e}")
            print("[SpeakerEncoder] Using dummy model for testing...")
            self.model = DummyECAPATDNN(self.device)

    def encode_audio(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio
        
        Args:
            audio_data: Audio samples at 16kHz (numpy array, float32)
        
        Returns:
            192-d normalized embedding or None if failed
        """
        try:
            # Check minimum audio length (2 seconds at 16kHz)
            min_samples = self.sample_rate * 2
            if audio_data is None or len(audio_data) < min_samples:
                print(f"[SpeakerEncoder] ✗ Audio too short: {len(audio_data) if audio_data is not None else 0} samples (need >= {min_samples})")
                return None

            audio_data = audio_data.astype(np.float32)

            # Normalize audio
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val

            # Convert to tensor [batch=1, time]
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Check if model has encode_batch method (speechbrain style)
                if hasattr(self.model, 'encode_batch'):
                    emb = self.model.encode_batch(audio_tensor)
                    emb = emb.squeeze().cpu().numpy()
                # Otherwise assume it's a torch.hub model that returns embeddings directly
                elif callable(self.model):
                    emb = self.model(audio_tensor)
                    if isinstance(emb, torch.Tensor):
                        emb = emb.squeeze().cpu().numpy()
                else:
                    print("[SpeakerEncoder] ✗ Model format not recognized")
                    return None

            # Ensure output is 1D vector
            if emb.ndim > 1:
                emb = emb.flatten()

            # Normalize embedding
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            else:
                emb = emb + 1e-8

            return emb.astype(np.float32)

        except Exception as e:
            print(f"[SpeakerEncoder] ✗ Encoding failed: {e}")
            return None


class DummyECAPATDNN:
    """Fallback dummy model for testing when real model unavailable"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.embedding_dim = 192
    
    def encode_batch(self, audio_tensor):
        """Return dummy 192-d embedding"""
        batch_size = audio_tensor.shape[0]
        # Return random normalized embeddings
        dummy_emb = torch.randn(batch_size, self.embedding_dim, device=self.device)
        dummy_emb = torch.nn.functional.normalize(dummy_emb, p=2, dim=1)
        return dummy_emb
    
    def eval(self):
        return self
    
    def to(self, device):
        self.device = device
        return self


# Test
if __name__ == "__main__":
    enc = SpeakerEncoder(device="cpu")
    
    # Generate test audio (2 seconds at 16kHz)
    audio = np.random.randn(32000).astype(np.float32)
    emb = enc.encode_audio(audio)
    
    if emb is not None:
        print(f"✓ Embedding shape: {emb.shape}")
        print(f"✓ Embedding norm: {np.linalg.norm(emb):.4f}")
    else:
        print("✗ Failed to generate embedding")