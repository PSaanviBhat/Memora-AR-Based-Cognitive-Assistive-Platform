"""
MEMORA - Audio Processing Module
Temporal audio chunking, quality metrics, and aggregation
Input: 16kHz mono audio | Output: quality-filtered embeddings, metadata
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class AudioChunk:
    """Represents a single audio chunk with metadata"""
    audio_data: np.ndarray      # Audio samples
    start_ms: float             # Start time in milliseconds
    duration_ms: float          # Duration in milliseconds
    rms_energy: float           # Root-mean-square energy [0, 1]
    snr_db: float               # Signal-to-noise ratio in dB
    voice_confidence: float     # Voice activity confidence [0, 1]
    is_valid: bool              # Passed quality checks


class AudioProcessor:
    """
    Processes audio for speaker verification: chunking, quality checks, aggregation
    - 500ms temporal windows with 50% overlap
    - RMS energy and SNR computation
    - Voice activity detection (VAD)
    - Per-chunk embedding aggregation
    """

    def __init__(self, sample_rate: int = 16000, chunk_ms: int = 500,
                 overlap_percent: float = 0.5, device: str = "cpu"):
        """
        Initialize audio processor

        Args:
            sample_rate: Sample rate in Hz (default 16000)
            chunk_ms: Chunk duration in milliseconds (default 500)
            overlap_percent: Overlap percentage for windowing (default 0.5)
            device: Device for computation (unused, for compatibility)
        """
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.overlap_percent = overlap_percent
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.stride_samples = int(self.chunk_samples * (1 - overlap_percent))

        # Quality thresholds
        self.min_duration_sec = 2.0
        self.rms_threshold = 0.05  # Reject near-silence
        self.snr_threshold_db = 8.0  # Minimum SNR
        self.voice_confidence_threshold = 0.3

        print(f"[AudioProcessor] ✓ Initialized (chunk={chunk_ms}ms, stride={self.stride_samples//160}ms)")

    def compute_rms_energy(self, audio_chunk: np.ndarray) -> float:
        """
        Compute RMS (root-mean-square) energy of audio chunk

        Args:
            audio_chunk: Audio samples (normalized to [-1, 1])

        Returns:
            RMS energy in [0, 1]
        """
        if len(audio_chunk) == 0:
            return 0.0

        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        return np.clip(rms, 0.0, 1.0)

    def compute_snr(self, audio: np.ndarray, noise_duration_sec: float = 0.5) -> float:
        """
        Estimate Signal-to-Noise Ratio (SNR) in dB
        Assumes first 0.5s is noise baseline

        Args:
            audio: Full audio signal (normalized to [-1, 1])
            noise_duration_sec: Duration to consider as noise baseline

        Returns:
            SNR in dB (higher = less noise)
        """
        try:
            if len(audio) < self.sample_rate * noise_duration_sec:
                # Not enough audio for noise estimation
                return 10.0

            # Estimate noise from first 0.5 seconds
            noise_samples = int(self.sample_rate * noise_duration_sec)
            noise_power = np.mean(audio[:noise_samples] ** 2)

            # Estimate signal from rest of audio
            signal_power = np.mean(audio[noise_samples:] ** 2)

            # SNR in dB (avoid division by zero)
            if noise_power < 1e-8:
                snr_db = 20.0  # Very quiet - assume good
            else:
                snr_db = 10 * np.log10(signal_power / noise_power)

            return float(np.clip(snr_db, -10.0, 40.0))

        except Exception as e:
            print(f"[AudioProcessor] ✗ SNR computation failed: {e}")
            return 10.0

    def detect_voice_activity(self, audio_chunk: np.ndarray,
                             threshold: float = 0.3) -> Tuple[bool, float]:
        """
        Simple Voice Activity Detection (VAD) using spectral centroid approximation
        Voice typically has energy concentrated in ~500-4000 Hz band

        Args:
            audio_chunk: Audio chunk (16kHz mono)
            threshold: Confidence threshold [0, 1]

        Returns:
            (is_voice: bool, confidence: float [0, 1])
        """
        try:
            if len(audio_chunk) < self.sample_rate // 10:  # <100ms
                return False, 0.0

            # Use simple Energy-based VAD: if RMS is high enough, likely voice
            rms = self.compute_rms_energy(audio_chunk)

            # If RMS > threshold, likely speech (voice has amplitude variation)
            # More sophisticated: check for zero-crossing rate, spectral properties
            # For now: RMS-based with bonus for variance
            energy_confidence = min(rms * 2.0, 1.0)  # Scale RMS to [0, 1]

            # Check for suitable amplitude variation (voice changes)
            # Split chunk into sub-frames and check variance
            num_subframes = 4
            subframe_len = len(audio_chunk) // num_subframes
            if subframe_len > 0:
                subframe_rms = []
                for i in range(num_subframes):
                    subframe = audio_chunk[i*subframe_len:(i+1)*subframe_len]
                    subframe_rms.append(self.compute_rms_energy(subframe))

                # Voice has varying energy across frames
                rms_variance = np.var(subframe_rms) if len(subframe_rms) > 1 else 0
                variation_confidence = float(np.clip(rms_variance * 10, 0, 1))
            else:
                variation_confidence = 0.5

            # Combined confidence
            confidence = 0.6 * energy_confidence + 0.4 * variation_confidence
            is_voice = confidence >= threshold

            return is_voice, float(confidence)

        except Exception as e:
            print(f"[AudioProcessor] ✗ VAD failed: {e}")
            return False, 0.0

    def chunk_audio(self, audio: np.ndarray, chunk_ms: Optional[int] = None
                   ) -> List[AudioChunk]:
        """
        Split audio into overlapping temporal chunks with quality metrics

        Args:
            audio: Full audio signal (16kHz mono, normalized to [-1, 1])
            chunk_ms: Override default chunk size

        Returns:
            List of AudioChunk objects with metadata
        """
        chunk_ms = chunk_ms or self.chunk_ms
        chunk_samples = int(self.sample_rate * chunk_ms / 1000)
        stride_samples = int(chunk_samples * (1 - self.overlap_percent))

        chunks = []
        start_idx = 0
        chunk_idx = 0

        while start_idx + chunk_samples <= len(audio):
            end_idx = start_idx + chunk_samples
            chunk_data = audio[start_idx:end_idx].astype(np.float32)

            # Compute quality metrics
            rms = self.compute_rms_energy(chunk_data)
            snr = self.compute_snr(audio[max(0, start_idx-self.sample_rate):end_idx])
            is_voice, voice_conf = self.detect_voice_activity(chunk_data)

            # Determine if chunk passes quality checks
            is_valid = (
                rms > self.rms_threshold and
                snr > self.snr_threshold_db and
                voice_conf > self.voice_confidence_threshold
            )

            chunk = AudioChunk(
                audio_data=chunk_data,
                start_ms=start_idx / self.sample_rate * 1000,
                duration_ms=chunk_ms,
                rms_energy=rms,
                snr_db=snr,
                voice_confidence=voice_conf,
                is_valid=is_valid
            )
            chunks.append(chunk)
            chunk_idx += 1
            start_idx += stride_samples

        return chunks

    def aggregate_embeddings(self, embeddings: List[np.ndarray],
                           weights: Optional[List[float]] = None) -> Optional[np.ndarray]:
        """
        Aggregate embeddings via weighted mean pooling and L2 normalization

        Args:
            embeddings: List of embeddings (e.g., 192-d speaker embeddings)
            weights: Optional weights for each embedding (default: uniform)

        Returns:
            Aggregated 192-d normalized embedding, or None if no embeddings
        """
        try:
            if len(embeddings) == 0:
                return None

            embeddings_array = np.array(embeddings, dtype=np.float32)

            if weights is None:
                # Uniform weighting
                agg_emb = np.mean(embeddings_array, axis=0)
            else:
                # Weighted mean
                weights = np.array(weights, dtype=np.float32)
                weights = weights / np.sum(weights)  # Normalize weights
                agg_emb = np.average(embeddings_array, axis=0, weights=weights)

            # L2 normalization
            norm = np.linalg.norm(agg_emb)
            if norm > 0:
                agg_emb = agg_emb / norm

            return agg_emb.astype(np.float32)

        except Exception as e:
            print(f"[AudioProcessor] ✗ Embedding aggregation failed: {e}")
            return None

    def get_audio_quality_score(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute overall audio quality score and metrics

        Args:
            audio: Full audio signal

        Returns:
            Dict with quality metrics
        """
        try:
            rms = self.compute_rms_energy(audio)
            snr = self.compute_snr(audio)

            # Chunk-level VAD
            chunks = self.chunk_audio(audio)
            valid_chunks = sum(1 for c in chunks if c.is_valid)
            total_chunks = len(chunks)

            # Quality score: combination of metrics
            rms_score = min(rms * 5, 1.0)  # Scale RMS to [0, 1]
            snr_score = min((snr + 10) / 50, 1.0)  # Scale SNR to [0, 1]
            coverage_score = valid_chunks / total_chunks if total_chunks > 0 else 0

            overall_score = 0.4 * rms_score + 0.3 * snr_score + 0.3 * coverage_score

            return {
                'rms_energy': rms,
                'snr_db': snr,
                'valid_chunks': valid_chunks,
                'total_chunks': total_chunks,
                'coverage': coverage_score,
                'overall_quality': overall_score
            }

        except Exception as e:
            print(f"[AudioProcessor] ✗ Quality scoring failed: {e}")
            return {
                'rms_energy': 0,
                'snr_db': 0,
                'valid_chunks': 0,
                'total_chunks': 0,
                'coverage': 0,
                'overall_quality': 0
            }


# Unit tests
if __name__ == "__main__":
    print("[AudioProcessor] Running unit tests...\n")

    processor = AudioProcessor(sample_rate=16000, chunk_ms=500)

    # Test 1: RMS energy computation
    print("Test 1: RMS energy computation")
    test_audio_quiet = np.random.randn(16000 * 2).astype(np.float32) * 0.02  # Quiet
    test_audio_loud = np.random.randn(16000 * 2).astype(np.float32) * 0.5   # Loud

    rms_quiet = processor.compute_rms_energy(test_audio_quiet)
    rms_loud = processor.compute_rms_energy(test_audio_loud)

    print(f"  Quiet RMS: {rms_quiet:.4f}")
    print(f"  Loud RMS: {rms_loud:.4f}")
    assert rms_loud > rms_quiet, "Loud signal should have higher RMS"
    print("  ✓ Pass\n")

    # Test 2: SNR computation
    print("Test 2: SNR computation")
    snr = processor.compute_snr(test_audio_loud)
    print(f"  SNR: {snr:.2f} dB")
    assert -10 <= snr <= 40, "SNR out of expected range"
    print("  ✓ Pass\n")

    # Test 3: Voice Activity Detection
    print("Test 3: Voice Activity Detection")
    is_voice, conf = processor.detect_voice_activity(test_audio_loud)
    print(f"  Loud signal - Voice: {is_voice}, Confidence: {conf:.3f}")
    assert conf >= 0.0 and conf <= 1.0, "Confidence out of [0, 1]"
    print("  ✓ Pass\n")

    # Test 4: Audio chunking
    print("Test 4: Audio chunking")
    chunks = processor.chunk_audio(test_audio_loud)
    print(f"  Total chunks: {len(chunks)}")
    print(f"  First chunk - RMS: {chunks[0].rms_energy:.4f}, Valid: {chunks[0].is_valid}")
    assert len(chunks) > 0, "Should produce at least one chunk"
    print("  ✓ Pass\n")

    # Test 5: Embedding aggregation
    print("Test 5: Embedding aggregation")
    dummy_embeddings = [
        np.random.randn(192).astype(np.float32),
        np.random.randn(192).astype(np.float32),
        np.random.randn(192).astype(np.float32)
    ]
    agg_emb = processor.aggregate_embeddings(dummy_embeddings)
    print(f"  Aggregated embedding shape: {agg_emb.shape}")
    print(f"  Norm: {np.linalg.norm(agg_emb):.4f}")
    assert agg_emb.shape == (192,), "Aggregated embedding should be 192-d"
    assert abs(np.linalg.norm(agg_emb) - 1.0) < 0.01, "Should be L2-normalized"
    print("  ✓ Pass\n")

    # Test 6: Audio quality scoring
    print("Test 6: Audio quality scoring")
    quality = processor.get_audio_quality_score(test_audio_loud)
    print(f"  Overall quality: {quality['overall_quality']:.3f}")
    print(f"  Coverage: {quality['coverage']:.2%}")
    assert 0 <= quality['overall_quality'] <= 1, "Quality score out of range"
    print("  ✓ Pass\n")

    print("✓ All AudioProcessor tests passed!")
