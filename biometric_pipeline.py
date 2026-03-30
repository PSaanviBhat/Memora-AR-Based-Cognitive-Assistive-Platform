"""
MEMORA - Biometric Pipeline
Main orchestrator: Camera + Audio → Face/Voice embeddings → IC fusion → Identity table
Interactive CLI for registration and verification
WITH: Audio quality filtering, face quality filtering, async STT integration
"""
import torch
import cv2
import numpy as np
import threading
import time
import json
from typing import Optional, Dict
import warnings
import os

# Fix macOS camera authorization issue
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except Exception:
    pyaudio = None
    PYAUDIO_AVAILABLE = False

warnings.filterwarnings('ignore')

from face_encoder import FaceEncoder
from speaker_encoder import SpeakerEncoder
from identity_confidence import IdentityConfusionFuser, to_json
from identity_table import IdentityTable
from audio_processor import AudioProcessor
from face_processor import FaceQualityFilter
from stt_handler import WhisperSTTHandler


class BiometricPipeline:
    """
    Main T1 system: Orchestrates all components
    - Parallel camera + audio capture with quality filtering
    - Face/voice embedding extraction
    - Identity confidence fusion
    - SQLite storage
    - Async STT integration
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize pipeline

        Args:
            device: "cuda" or "cpu"
        """
        print("[BiometricPipeline] Initializing...")
        self.device = device

        # Load models
        self.face_encoder = FaceEncoder(device=device)
        self.speaker_encoder = SpeakerEncoder(device=device)
        self.ic_fuser = IdentityConfusionFuser(
            fusion_weight=0.6,
            theta_known=0.70,
            theta_high=0.85,
            theta_medium=0.70
        )
        self.identity_table = IdentityTable()

        # NEW: Quality filtering modules
        self.audio_processor = AudioProcessor(sample_rate=16000, chunk_ms=500)
        self.face_filter = FaceQualityFilter(confidence_threshold=0.95, pose_limits_deg=(30, 30, 30))
        self.stt_handler = WhisperSTTHandler(model="tiny", device=device)

        # State
        self.running = False
        self.face_embeddings = []
        self.face_quality_scores = []  # NEW: track quality scores
        self.audio_buffer = []
        self.audio_level = 0.0
        self.camera_error = False
        self.transcript = None  # NEW: STT result storage
        self.stt_thread = None  # NEW: STT thread reference

        # Audio config
        self.audio_format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        self.audio_channels = 1
        self.audio_rate = 16000
        self.audio_chunk = 1024

        if not PYAUDIO_AVAILABLE:
            print("[BiometricPipeline] ! PyAudio not installed: microphone capture disabled")

        print("[BiometricPipeline] ✓ Initialized successfully\n")

    def _on_transcript_complete(self, result: Optional[Dict]):
        """Callback for async STT completion"""
        if result:
            self.transcript = result
            text = result.get('text', '[no text]')
            print(f"[BiometricPipeline] ✓ Transcript: \"{text[:60]}...\"")
        else:
            print("[BiometricPipeline] ! STT failed or timed out")

    def _validate_embeddings(self, face_emb: Optional[np.ndarray],
                            voice_emb: Optional[np.ndarray],
                            log: bool = True) -> bool:
        """
        Validate embedding shapes, NaN/Inf values, and normalization

        Args:
            face_emb: Face embedding (should be 128-d)
            voice_emb: Voice embedding (should be 192-d)
            log: Print validation errors

        Returns:
            True if valid, False otherwise
        """
        issues = []

        # Face validation
        if face_emb is None:
            issues.append("face_emb is None")
        elif face_emb.shape != (128,):
            issues.append(f"face_emb shape {face_emb.shape} != (128,)")
        elif np.any(np.isnan(face_emb)) or np.any(np.isinf(face_emb)):
            issues.append(f"face_emb contains NaN/Inf")
        elif not (0.99 <= np.linalg.norm(face_emb) <= 1.01):
            issues.append(f"face_emb not L2-normalized (norm={np.linalg.norm(face_emb):.4f})")

        # Voice validation
        if voice_emb is None:
            issues.append("voice_emb is None")
        elif voice_emb.shape != (192,):
            issues.append(f"voice_emb shape {voice_emb.shape} != (192,)")
        elif np.any(np.isnan(voice_emb)) or np.any(np.isinf(voice_emb)):
            issues.append(f"voice_emb contains NaN/Inf")
        elif not (0.99 <= np.linalg.norm(voice_emb) <= 1.01):
            issues.append(f"voice_emb not L2-normalized (norm={np.linalg.norm(voice_emb):.4f})")

        if issues and log:
            print(f"[BiometricPipeline] ✗ Embedding validation failed:")
            for issue in issues:
                print(f"  → {issue}")

        return len(issues) == 0

    def register_user(self, user_name: str, duration_sec: int = 10) -> bool:
        """
        Register new user: capture face + voice, extract embeddings, store

        Args:
            user_name: User's name
            duration_sec: Capture duration (default 10s)

        Returns:
            True if successful
        """
        print(f"\n{'='*60}")
        print(f"REGISTERING NEW USER: {user_name}")
        print(f"{'='*60}")
        print(f"Duration: {duration_sec}s")
        print(f"Webcam: Look at camera, keep face centered")
        print(f"Audio: Speak clearly about yourself")
        print(f"\nStarting in 3 seconds...")
        time.sleep(3)

        # Reset buffers
        self.running = True
        self.face_embeddings = []
        self.face_quality_scores = []
        self.audio_buffer = []
        self.camera_error = False
        self.transcript = None

        # Start capture threads
        cam_thread = threading.Thread(target=self._camera_capture, args=(duration_sec,), daemon=False)
        aud_thread = threading.Thread(target=self._audio_capture, args=(duration_sec,), daemon=False)

        cam_thread.start()
        aud_thread.start()

        # Wait with timeout
        cam_thread.join(timeout=duration_sec + 5)
        aud_thread.join(timeout=duration_sec + 5)

        self.running = False
        time.sleep(0.5)

        # Process embeddings
        print("\n[Processing] Averaging face embeddings (quality-weighted)...")
        face_emb = self._average_face_embeddings()

        print("[Processing] Extracting speaker embedding (aggregated)...")
        audio_data = self._get_audio_data()
        if audio_data is not None:
            voice_emb = self.speaker_encoder.encode_audio(audio_data)
        else:
            voice_emb = None

        # Validate
        if face_emb is None:
            print("✗ Registration FAILED: No face detected")
            return False

        if voice_emb is None:
            print("✗ Registration FAILED: No voice detected")
            return False

        if not self._validate_embeddings(face_emb, voice_emb):
            print("✗ Registration FAILED: Embedding validation errors")
            return False

        print(f"✓ Face samples: {len(self.face_embeddings)}")
        print(f"✓ Audio duration: {len(audio_data) / self.audio_rate:.1f}s")

        # Wait for STT with timeout
        stt_result = None
        if self.stt_thread:
            self.stt_thread.join(timeout=3.0)
            stt_result = self.transcript

        # Compute IC and store
        user_id = f"{user_name}_{int(time.time())}"
        ic = self.ic_fuser.compute_ic(user_id, face_emb, voice_emb)

        metadata = {
            'confidence_level': ic.confidence_level,
            'registered_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'transcript': stt_result.get('text', '') if stt_result else '',
            'face_quality_scores': self.face_quality_scores.tolist() if self.face_quality_scores else [],
            'audio_quality': self.audio_processor.get_audio_quality_score(audio_data) if audio_data is not None else {}
        }

        success = self.identity_table.add_identity(
            user_id=user_id,
            name=user_name,
            face_emb=face_emb,
            voice_emb=voice_emb,
            ic_score=ic.IC_score,
            metadata=metadata
        )

        if success:
            print(f"\n{'='*60}")
            print(f"✓ REGISTRATION SUCCESSFUL")
            print(f"{'='*60}")
            print(f"User: {user_name}")
            print(f"ID: {user_id}")
            print(f"\nIdentity Confidence Output (JSON for T3):")
            print(json.dumps(to_json(ic), indent=2))
            print(f"{'='*60}\n")
        else:
            print("✗ Failed to store identity")
            return False

        return success

    def verify_user(self, duration_sec: int = 5) -> Optional[Dict]:
        """
        Verify user identity: capture face + voice, compute IC, search stored users

        Args:
            duration_sec: Capture duration (default 5s)

        Returns:
            JSON output dict or None if failed
        """
        print(f"\n{'='*60}")
        print(f"VERIFYING IDENTITY")
        print(f"{'='*60}")
        print(f"Duration: {duration_sec}s")
        print(f"Webcam: Look at camera")
        print(f"Audio: Say something")
        print(f"\nStarting in 3 seconds...")
        time.sleep(3)

        # Reset buffers
        self.running = True
        self.face_embeddings = []
        self.face_quality_scores = []
        self.audio_buffer = []
        self.camera_error = False
        self.transcript = None

        # Start capture threads
        cam_thread = threading.Thread(target=self._camera_capture, args=(duration_sec,), daemon=False)
        aud_thread = threading.Thread(target=self._audio_capture, args=(duration_sec,), daemon=False)

        cam_thread.start()
        aud_thread.start()

        # Wait with timeout
        cam_thread.join(timeout=duration_sec + 5)
        aud_thread.join(timeout=duration_sec + 5)

        self.running = False
        time.sleep(0.5)

        # Process embeddings
        print("\n[Processing] Averaging face embeddings...")
        face_emb = self._average_face_embeddings()

        print("[Processing] Extracting speaker embedding...")
        audio_data = self._get_audio_data()
        if audio_data is not None:
            voice_emb = self.speaker_encoder.encode_audio(audio_data)
        else:
            voice_emb = None

        # Validate
        if face_emb is None or voice_emb is None:
            print("✗ Verification FAILED: Insufficient biometric data")
            return None

        if not self._validate_embeddings(face_emb, voice_emb):
            print("✗ Verification FAILED: Embedding validation errors")
            return None

        # Search stored identities
        stored_identities = self.identity_table.list_all()

        if len(stored_identities) == 0:
            print("✗ No registered users to match against")
            return None

        print(f"\n[Searching] {len(stored_identities)} registered users...")

        best_match = None
        best_score = -1.0
        results = []

        for stored_user in stored_identities:
            stored_data = self.identity_table.get_identity(stored_user['user_id'])
            stored_face = stored_data['face_vector']
            stored_voice = stored_data['voice_vector']

            # Compute similarity with this user
            face_sim = np.dot(face_emb, stored_face)
            voice_sim = np.dot(voice_emb, stored_voice)
            combined_sim = 0.6 * face_sim + 0.4 * voice_sim

            results.append({
                'name': stored_user['name'],
                'face_sim': face_sim,
                'voice_sim': voice_sim,
                'combined': combined_sim
            })

            if combined_sim > best_score:
                best_score = combined_sim
                best_match = stored_user

        # Display top matches
        print("\nTop matches:")
        for i, r in enumerate(sorted(results, key=lambda x: x['combined'], reverse=True)[:3]):
            print(f"  {i+1}. {r['name']}: {r['combined']:.4f} (face: {r['face_sim']:.4f}, voice: {r['voice_sim']:.4f})")

        if best_match is None:
            print("\n✗ No match found")
            return None

        # Compute IC for best match
        best_data = self.identity_table.get_identity(best_match['user_id'])
        best_face_stored = best_data['face_vector']
        best_voice_stored = best_data['voice_vector']

        ic = self.ic_fuser.compute_ic(
            best_match['user_id'],
            face_emb,
            voice_emb,
            face_ref_emb=best_face_stored,
            voice_ref_emb=best_voice_stored,
        )

        # Update last_seen
        self.identity_table.update_last_seen(best_match['user_id'])

        json_out = to_json(ic)

        print(f"\n{'='*60}")
        print(f"✓ BEST MATCH: {best_match['name']}")
        print(f"{'='*60}")
        print(f"\nIdentity Confidence Output (JSON for T3):")
        print(json.dumps(json_out, indent=2))
        print(f"{'='*60}\n")

        return json_out

    def list_users(self):
        """Display all registered users"""
        users = self.identity_table.list_all()

        if len(users) == 0:
            print("\n✗ No registered users\n")
            return

        print(f"\n{'='*60}")
        print(f"REGISTERED USERS ({len(users)})")
        print(f"{'='*60}")

        for i, user in enumerate(users, 1):
            print(f"{i}. {user['name']}")
            print(f"   ID: {user['user_id']}")
            print(f"   Registered: {user['registered_at']}")
            print(f"   IC Score: {user['ic_score']:.4f}\n")

        print(f"{'='*60}\n")

    def _camera_capture(self, duration_sec: int):
        """Background thread: capture video from webcam with quality filtering"""
        try:
            print("[Camera] Initializing...")
            cap = cv2.VideoCapture(0)

            # Set properties BEFORE checking isOpened()
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print("✗ Cannot open webcam")
                self.camera_error = True
                return

            print("[Camera] ✓ Opened successfully")

            start = time.time()
            frame_count = 0
            low_quality_rejected = 0

            while self.running and (time.time() - start < duration_sec):
                ret, frame = cap.read()

                if not ret:
                    time.sleep(0.01)
                    continue

                # Get faces from frame
                faces = self.face_encoder.model.get(frame)

                if len(faces) == 0:
                    frame_count += 1
                    continue

                # NEW: Filter by quality and select best face
                result = self.face_filter.filter_faces(faces, frame, min_quality=0.5)

                if result is None:
                    low_quality_rejected += 1
                    frame_count += 1
                    continue

                best_face, quality_score, metrics = result

                # Extract embedding from best face
                embedding = best_face.embedding / np.linalg.norm(best_face.embedding)
                embedding = embedding.astype(np.float32)

                self.face_embeddings.append(embedding)
                self.face_quality_scores.append(quality_score)

                print(f"  ✓ Face {len(self.face_embeddings)} (quality: {quality_score:.3f}, "
                      f"det: {metrics.get('det_score', 0):.3f})")

                frame_count += 1
                elapsed = time.time() - start

                if frame_count % 5 == 0:
                    status = f"Faces: {len(self.face_embeddings)} | Rejected: {low_quality_rejected} | Time: {elapsed:.1f}s/{duration_sec}s"
                    print(f"  {status}")

            cap.release()
            print(f"\n[Camera] ✓ Captured {len(self.face_embeddings)} embeddings from {frame_count} frames "
                  f"({low_quality_rejected} rejected)")

        except Exception as e:
            print(f"✗ Camera error: {e}")
            import traceback
            traceback.print_exc()
            self.camera_error = True

    def _audio_capture(self, duration_sec: int):
        """Background thread: capture audio from microphone with STT"""
        if not PYAUDIO_AVAILABLE:
            print("[Microphone] ! PyAudio not available, skipping audio capture")
            return

        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=self.audio_format,
                          channels=self.audio_channels,
                          rate=self.audio_rate,
                          input=True,
                          frames_per_buffer=self.audio_chunk)

            start = time.time()
            chunk_count = 0

            print("[Microphone] ✓ Recording started...")

            while self.running and (time.time() - start < duration_sec):
                try:
                    data = stream.read(self.audio_chunk, exception_on_overflow=False)
                    self.audio_buffer.append(data)
                    chunk_count += 1

                    # Compute audio level
                    audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    self.audio_level = np.sqrt(np.mean(audio_array ** 2)) / 32768.0

                    elapsed = time.time() - start
                    if chunk_count % 8 == 0:
                        level_bars = "█" * int(self.audio_level * 50)
                        print(f"  {elapsed:.1f}s: {level_bars} ({self.audio_level:.3f})")

                except Exception as e:
                    print(f"  Audio read error: {e}")
                    continue

            stream.stop_stream()
            stream.close()
            p.terminate()

            total_duration = chunk_count * self.audio_chunk / self.audio_rate
            print(f"[Microphone] ✓ Captured {len(self.audio_buffer)} chunks ({total_duration:.1f}s)")

            # NEW: Launch async STT on captured audio
            if len(self.audio_buffer) > 0:
                audio_data = self._get_audio_data()
                if audio_data is not None and len(audio_data) >= self.audio_rate * 2:
                    self.stt_thread = self.stt_handler.transcribe_audio_async(
                        audio_data,
                        callback=self._on_transcript_complete,
                        timeout_sec=10.0
                    )

        except Exception as e:
            print(f"✗ Microphone error: {e}")

    def _average_face_embeddings(self) -> Optional[np.ndarray]:
        """Average face embeddings from buffer (quality-weighted)"""
        if len(self.face_embeddings) < 3:
            print(f"✗ Insufficient face samples: {len(self.face_embeddings)} (need >= 3)")
            if self.camera_error:
                print("  → Camera failed to initialize. Check permissions.")
            return None

        # Quality-weighted average
        embeddings_array = np.array(self.face_embeddings, dtype=np.float32)
        if len(self.face_quality_scores) > 0:
            scores = np.array(self.face_quality_scores, dtype=np.float32)
            scores = scores / np.sum(scores)  # Normalize weights
            avg_emb = np.average(embeddings_array, axis=0, weights=scores)
        else:
            avg_emb = np.mean(embeddings_array, axis=0)

        # L2 normalization
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb = avg_emb / norm

        return avg_emb.astype(np.float32)

    def _get_audio_data(self) -> Optional[np.ndarray]:
        """Convert audio buffer to numpy array"""
        if len(self.audio_buffer) == 0:
            print("✗ No audio data captured")
            return None

        try:
            audio_data = b''.join(self.audio_buffer)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_array
        except Exception as e:
            print(f"✗ Audio conversion error: {e}")
            return None


def main():
    """Interactive CLI menu"""
    print("\n" + "="*60)
    print("MEMORA - Biometric Identity System (T1)")
    print("="*60)
    print("Identity & Biometrics Module")
    print("Face Recognition (ArcFace) + Speaker Verification (ECAPA-TDNN)")
    print("WITH: Audio quality + Face quality + Async STT (Whisper)")
    print("="*60 + "\n")

    # Initialize pipeline
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[MEMORA] Using device: {device}\n")
        pipeline = BiometricPipeline(device=device)
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return

    # Main loop
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("[1] Register new user")
        print("[2] Verify identity")
        print("[3] List registered users")
        print("[4] Exit")
        print("="*60)

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            user_name = input("Enter user name: ").strip()
            if not user_name:
                print("✗ Name cannot be empty")
                continue

            duration = input("Duration (seconds, default 10): ").strip()
            try:
                duration = int(duration) if duration else 10
            except ValueError:
                duration = 10

            pipeline.register_user(user_name, duration_sec=duration)

        elif choice == "2":
            pipeline.verify_user(duration_sec=5)

        elif choice == "3":
            pipeline.list_users()

        elif choice == "4":
            print("\n✓ Exiting...")
            break

        else:
            print("✗ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
