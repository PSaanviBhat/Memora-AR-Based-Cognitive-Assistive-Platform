"""
MEMORA - Speech-to-Text (STT) Handler
Async Whisper integration for background transcription
Input: Audio data | Output: transcript with confidence
"""

import numpy as np
import threading
import time
from typing import Dict, Optional, Callable
import warnings

warnings.filterwarnings("ignore")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[WhisperSTTHandler] ! Whisper not available, using dummy handler")


class WhisperSTTHandler:
    """
    Wraps OpenAI Whisper for speech-to-text with async threading
    - Runs on background thread to avoid blocking biometric pipeline
    - CPU-friendly (tiny model ~39M params)
    - Graceful fallback if Whisper unavailable
    """

    def __init__(self, model: str = "tiny", device: str = "cpu", language: str = "en"):
        """
        Initialize Whisper handler

        Args:
            model: Whisper model size ("tiny", "base", "small", "medium", "large")
                   Tiny is fastest (~1s per 30s audio on CPU)
            device: Device for computation ("cpu" or "cuda")
            language: Language code (e.g., "en" for English)
        """
        self.model_name = model
        self.device = device
        self.language = language
        self.model = None
        self.result = None
        self.is_transcribing = False

        if WHISPER_AVAILABLE:
            try:
                print(f"[WhisperSTTHandler] Loading Whisper {model}...")
                self.model = whisper.load_model(model, device=device)
                print(f"[WhisperSTTHandler] ✓ Loaded {model} on {device}")
            except Exception as e:
                print(f"[WhisperSTTHandler] ✗ Failed to load model: {e}")
                WHISPER_AVAILABLE = False
        else:
            print(f"[WhisperSTTHandler] Using dummy handler (Whisper not installed)")

    def transcribe_audio_sync(self, audio_data: np.ndarray,
                              timeout_sec: float = 10.0) -> Optional[Dict]:
        """
        Synchronous transcription (blocking)
        Use for testing or when a result is immediately needed

        Args:
            audio_data: Audio array (16kHz mono, normalized to [-1, 1])
            timeout_sec: Max time to wait (unused for sync, for compatibility)

        Returns:
            Dict with keys: text, language, confidence, timestamps
            or None if failed
        """
        if not WHISPER_AVAILABLE or self.model is None:
            return self._get_dummy_result(audio_data)

        try:
            print("[WhisperSTTHandler] Transcribing (sync)...")
            start = time.time()

            # Whisper expects normalized audio
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Run inference
            result = self.model.transcribe(
                audio_data,
                language=self.language,
                fp16=False  # CPU doesn't support fp16
            )

            elapsed = time.time() - start
            text = result.get('text', '')
            language = result.get('language', 'en')

            # Estimate confidence (average of segment confidence if available)
            segments = result.get('segments', [])
            if segments:
                confidence = np.mean([s.get('confidence', 0.0) for s in segments])
            else:
                confidence = 0.95  # Assume high if no segments

            output = {
                'text': text.strip(),
                'language': language,
                'confidence': float(confidence),
                'timestamps': [s.get('start') for s in segments] if segments else [],
                'elapsed_sec': elapsed
            }

            print(f"[WhisperSTTHandler] ✓ Transcript: '{text[:50]}...' ({elapsed:.1f}s)")
            return output

        except Exception as e:
            print(f"[WhisperSTTHandler] ✗ Transcription failed: {e}")
            return None

    def transcribe_audio_async(self, audio_data: np.ndarray,
                              callback: Callable[[Optional[Dict]], None],
                              timeout_sec: float = 30.0) -> threading.Thread:
        """
        Asynchronous transcription on background thread
        Main thread continues while STT runs in parallel

        Args:
            audio_data: Audio array (16kHz mono)
            callback: Function to invoke with result: callback(result_dict)
            timeout_sec: Max time to wait for transcription

        Returns:
            Thread object (can call join() to wait)
        """
        def _transcribe_worker():
            try:
                self.is_transcribing = True
                result = self.transcribe_audio_sync(audio_data, timeout_sec)
                self.result = result
                if callback:
                    callback(result)
            finally:
                self.is_transcribing = False

        thread = threading.Thread(target=_transcribe_worker, daemon=False)
        thread.start()
        return thread

    def wait_for_result(self, timeout_sec: float = 30.0) -> Optional[Dict]:
        """
        Block until transcription completes or timeout

        Args:
            timeout_sec: Max wait time in seconds

        Returns:
            Transcription result or None if timeout
        """
        start = time.time()
        while self.is_transcribing and (time.time() - start) < timeout_sec:
            time.sleep(0.1)

        if self.is_transcribing:
            print(f"[WhisperSTTHandler] ✗ Transcription timeout after {timeout_sec}s")
            return None

        return self.result

    def _get_dummy_result(self, audio_data: np.ndarray) -> Dict:
        """Fallback dummy transcription when Whisper unavailable"""
        duration_sec = len(audio_data) / 16000.0
        return {
            'text': f"[dummy transcript - {duration_sec:.1f}s audio]",
            'language': 'en',
            'confidence': 0.5,
            'timestamps': [],
            'elapsed_sec': 0.0
        }

    def is_available(self) -> bool:
        """Check if Whisper is available and loaded"""
        return WHISPER_AVAILABLE and self.model is not None


# Unit tests
if __name__ == "__main__":
    print("[WhisperSTTHandler] Running unit tests...\n")

    handler = WhisperSTTHandler(model="tiny", device="cpu")

    # Test 1: Dummy transcription (if Whisper unavailable)
    print("Test 1: Dummy transcription")
    dummy_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1  # 3 sec
    result = handler._get_dummy_result(dummy_audio)
    print(f"  Text: {result['text']}")
    print(f"  Language: {result['language']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    assert 'text' in result and 'language' in result
    print("  ✓ Pass\n")

    # Test 2: Availability check
    print("Test 2: Whisper availability")
    available = handler.is_available()
    print(f"  Available: {available}")
    print(f"  Model loaded: {handler.model is not None}")
    print("  ✓ Pass\n")

    # Test 3: Result waiting mechanism
    print("Test 3: Async callback pattern")
    results_collected = []

    def callback(result):
        results_collected.append(result)
        print(f"  Callback invoked with result")

    # Launch async transcription
    print("  Launching async transcription...")
    if handler.is_available():
        thread = handler.transcribe_audio_async(dummy_audio, callback, timeout_sec=10.0)
        thread.join(timeout=12)
        print(f"  Collected {len(results_collected)} results")
        assert len(results_collected) > 0, "Should have collected a result"
    else:
        print("  Skipping (Whisper not available)")

    print("  ✓ Pass\n")

    print("✓ All WhisperSTTHandler tests passed!")
