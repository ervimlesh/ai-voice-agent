"""
Advanced speaker verification service using SpeechBrain speaker embeddings.

Provides true voice fingerprinting for speaker identification and verification,
enabling accurate differentiation between multiple speakers even during
simultaneous speech.

Features:
- SpeechBrain ECAPA-TDNN speaker embeddings (high-quality voice fingerprints)
- Speaker verification (is this the same person?)
- Speaker identification (who is this person?)
- Confidence scoring for all operations
- CPU/GPU automatic fallback
- Production-grade error handling
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class SpeakerEmbedding:
    """Voice fingerprint for a speaker."""
    speaker_id: str
    embedding: np.ndarray  # High-dimensional speaker embedding (192D typically)
    turn_count: int = 1
    confidence_avg: float = 0.0
    pitch_mean: float = 0.0  # Hz
    energy_mean: float = 0.0  # dB

    # For persistent identification
    patient_label: Optional[str] = None  # Patient1, Patient2, etc.
    locked: bool = False  # If True, identity is locked


class SpeakerVerificationService:
    """
    Speaker verification and identification using neural speaker embeddings.

    Uses SpeechBrain's ECAPA-TDNN (state-of-the-art) for voice fingerprinting.

    Advantages over MFCC-based approach:
    - 512-dimensional embeddings vs 80-dimensional MFCC
    - Trained on large speaker datasets (better generalization)
    - Naturally robust to background noise and audio quality
    - Superior to hand-crafted features for speaker recognition
    - Can work with very short audio segments (< 1 second)
    """

    def __init__(self, settings=None):
        self.settings = settings or {}
        self._classifier = None
        self._speakers: Dict[str, SpeakerEmbedding] = {}
        self._next_id = 1
        self._next_patient_num = 1
        self.device = self._resolve_device()
        self._init_model()

    def _resolve_device(self) -> str:
        """Determine if GPU is available."""
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Speaker verification device: {device}")
            return device
        except Exception:
            return "cpu"

    def _init_model(self):
        """Load SpeechBrain speaker verification model."""
        try:
            from speechbrain.pretrained import SpeakerRecognition

            logger.info("Loading SpeechBrain ECAPA-TDNN speaker verification model...")

            # ECAPA-TDNN is the state-of-the-art speaker recognition model
            # Pre-trained on VoxCeleb1/2 (massive speaker datasets)
            self._classifier = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            logger.info("✅ SpeechBrain ECAPA-TDNN loaded (512D embeddings)")

        except Exception as e:
            logger.error(f"Failed to load SpeechBrain speaker model: {e}")
            logger.warning("Falling back to lightweight embedding extraction")
            self._classifier = None

    def reset(self):
        """Clear all registered speakers."""
        self._speakers.clear()
        self._next_id = 1
        self._next_patient_num = 1
        logger.info("Speaker verification registry reset")

    # ══════════════════════════════════════════════════════════════════════════════
    # SPEAKER IDENTIFICATION (Who is this person?)
    # ══════════════════════════════════════════════════════════════════════════════

    def identify_speaker(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        min_duration: float = 0.2
    ) -> Tuple[str, float, Optional[str]]:
        """
        Identify which registered speaker this audio belongs to.

        Args:
            audio: Audio waveform
            sr: Sample rate
            min_duration: Minimum audio duration for reliable identification

        Returns:
            (speaker_id, confidence, patient_label)
            - speaker_id: S1, S2, S3, etc. (internal ID)
            - confidence: 0-1, how confident we are in the match
            - patient_label: Patient1, Patient2, etc., or None if not yet assigned
        """
        if audio is None or len(audio) == 0:
            return self._register_new_speaker(None), 0.0, None

        # Check minimum duration
        duration = len(audio) / sr
        if duration < min_duration:
            logger.debug(f"Audio too short ({duration:.2f}s) for reliable identification")
            return self._register_new_speaker(audio), 0.5, None

        # Extract embedding
        embedding = self.get_embedding(audio, sr)
        if embedding is None or len(embedding) == 0:
            return self._register_new_speaker(audio), 0.3, None

        # Find best match among registered speakers
        best_id = None
        best_sim = -1.0
        best_confidence = 0.0

        for spk_id, speaker in self._speakers.items():
            sim = self._cosine_similarity(embedding, speaker.embedding)
            logger.debug(f"  Compare with {spk_id}: shape={speaker.embedding.shape}, sim={sim:.4f}")
            if sim > best_sim:
                best_sim = sim
                best_id = spk_id

        # Determine matching threshold
        # 0.5 = maybe same person, 0.65 = likely same, 0.75+ = very confident
        threshold = getattr(self.settings, "speaker_verification_threshold", 0.70)

        logger.debug(f"  Best match: {best_id} with sim={best_sim:.4f}, threshold={threshold}")

        if best_id is not None and best_sim >= threshold:
            speaker = self._speakers[best_id]
            # Update speaker's embedding (running average for stability)
            self._update_embedding(best_id, embedding, best_sim)
            logger.debug(f"  → Matched {best_id}")
            return best_id, best_sim, speaker.patient_label

        # No match - register as new speaker (pass audio, not embedding)
        logger.debug(f"  → Registering new speaker (sim {best_sim:.4f} < threshold {threshold})")
        new_id = self._register_new_speaker_with_embedding(embedding)
        return new_id, max(best_sim, 0.3), None

    def get_embedding(self, audio: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
        """
        Extract speaker embedding (voice fingerprint) from audio.

        Returns: 512-dimensional numpy array (or None on failure)
        """
        try:
            import torch

            if audio is None or len(audio) == 0:
                return None

            # Resample if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # Skip very short audio
            if len(audio) < 3200:  # < 0.2 seconds at 16kHz
                return None

            # Use SpeechBrain if available
            if self._classifier is not None:
                try:
                    # Convert to tensor
                    wav = torch.from_numpy(audio).float().unsqueeze(0)
                    if self.device == "cuda":
                        wav = wav.cuda()

                    # Extract embedding
                    with torch.no_grad():
                        embedding = self._classifier.encode_batch(wav)

                    # Convert to numpy and flatten
                    embedding = embedding.squeeze().cpu().numpy()

                    # Ensure 1D array
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()

                    # Normalize to unit length (standard for speaker embeddings)
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

                    return embedding

                except Exception as e:
                    logger.debug(f"SpeechBrain embedding failed: {e}")

            # Fallback to MFCC-based embedding
            return self._fallback_embedding(audio, sr)

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def _fallback_embedding(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Fallback embedding extraction using MFCC + prosody + spectral features.
        Less accurate than SpeechBrain but works when neural models unavailable.
        Improved for stability and consistency.
        """
        try:
            import librosa
            from scipy import signal

            # MFCC coefficients (13D) - stable feature
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_feat = np.concatenate([mfcc_mean, mfcc_std])  # 26D

            # Prosody features (5D)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                p = pitches[idx, t]
                if p > 0:
                    pitch_values.append(p)
            pitch_mean = np.mean(pitch_values) if pitch_values else 0.0
            pitch_std = np.std(pitch_values) if pitch_values else 0.0
            pitch_median = np.median(pitch_values) if pitch_values else 0.0

            # Energy features (5D)
            energy = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
            energy_max = np.max(energy)
            energy_min = np.min(energy)

            # RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            rms_mean = np.mean(rms)

            # Spectral features (8D)
            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
            spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0])
            zero_cross = np.mean(librosa.feature.zero_crossing_rate(audio)[0])

            # Temporal features
            onset_strength = np.mean(librosa.onset.onset_strength(y=audio, sr=sr))
            tempogram = librosa.feature.tempogram(y=audio, sr=sr)
            tempo_mean = np.mean(tempogram)

            # Harmonic/Percussive features (4D)
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_energy = np.mean(harmonic ** 2)
            percussive_energy = np.mean(percussive ** 2)
            harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-9)
            percussive_ratio = percussive_energy / (harmonic_energy + percussive_energy + 1e-9)

            # Combine all features (80D)
            embedding = np.concatenate([
                mfcc_feat,                              # 26D
                [pitch_mean, pitch_std, pitch_median, rms_mean, np.max(rms)],   # 5D
                [energy_mean, energy_std, energy_max, energy_min, energy_max - energy_min],  # 5D
                [spec_centroid, spec_rolloff, zero_cross, onset_strength, tempo_mean],  # 5D
                [harmonic_energy, percussive_energy, harmonic_ratio, percussive_ratio],  # 4D
                # Remaining 10D: use spectral characteristics
                np.percentile(energy.flatten(), [10, 20, 30, 40, 50, 60, 70, 80, 90, 95])  # 10D
            ])

            # Ensure exactly 80D
            if len(embedding) < 80:
                embedding = np.concatenate([embedding, np.zeros(80 - len(embedding))])
            elif len(embedding) > 80:
                embedding = embedding[:80]

            # Normalize to unit length
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

            return embedding

        except Exception as e:
            logger.debug(f"Fallback embedding failed: {e}")
            return None

    # ══════════════════════════════════════════════════════════════════════════════
    # SPEAKER VERIFICATION (Are these the same person?)
    # ══════════════════════════════════════════════════════════════════════════════

    def verify_speakers(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        sr: int = 16000
    ) -> Tuple[bool, float]:
        """
        Verify if two audio segments are from the same speaker.

        Returns:
            (is_same_speaker, confidence_score)
            - is_same_speaker: True if probably same person
            - confidence_score: 0-1, how confident in this decision
        """
        try:
            emb1 = self.get_embedding(audio1, sr)
            emb2 = self.get_embedding(audio2, sr)

            if emb1 is None or emb2 is None:
                return False, 0.0

            similarity = self._cosine_similarity(emb1, emb2)

            # Thresholds for speaker verification
            threshold = getattr(self.settings, "speaker_verification_threshold", 0.70)

            is_same = similarity >= threshold
            return is_same, similarity

        except Exception as e:
            logger.error(f"Speaker verification failed: {e}")
            return False, 0.0

    # ══════════════════════════════════════════════════════════════════════════════
    # PATIENT LABELING
    # ══════════════════════════════════════════════════════════════════════════════

    def assign_patient_label(self, speaker_id: str) -> str:
        """
        Assign a patient label to a speaker (Patient1, Patient2, etc.).
        """
        if speaker_id not in self._speakers:
            return f"Patient{self._next_patient_num}"

        speaker = self._speakers[speaker_id]
        if speaker.patient_label is None:
            speaker.patient_label = f"Patient{self._next_patient_num}"
            self._next_patient_num += 1

        return speaker.patient_label

    def get_patient_label(self, speaker_id: str) -> Optional[str]:
        """Get the patient label for a speaker (if assigned)."""
        if speaker_id in self._speakers:
            return self._speakers[speaker_id].patient_label
        return None

    # ══════════════════════════════════════════════════════════════════════════════
    # INTERNAL HELPER METHODS
    # ══════════════════════════════════════════════════════════════════════════════

    def _register_new_speaker(self, audio: Optional[np.ndarray]) -> str:
        """Register a new speaker from audio and assign an ID."""
        speaker_id = f"S{self._next_id}"
        self._next_id += 1

        embedding = None
        if audio is not None:
            embedding = self.get_embedding(audio, 16000)

        if embedding is None:
            # Fallback embedding with appropriate dimensionality
            embedding = np.random.randn(80)  # Match fallback embedding size
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        else:
            # Ensure embedding is 1D numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.astype(np.float32)
                if embedding.ndim > 1:
                    embedding = embedding.flatten()

        speaker = SpeakerEmbedding(
            speaker_id=speaker_id,
            embedding=embedding,
            turn_count=1,
            confidence_avg=0.5
        )

        self._speakers[speaker_id] = speaker
        logger.info(f"✨ Registered new speaker {speaker_id}")

        return speaker_id

    def _register_new_speaker_with_embedding(self, embedding: np.ndarray) -> str:
        """Register a new speaker with a pre-extracted embedding and assign an ID."""
        speaker_id = f"S{self._next_id}"
        self._next_id += 1

        if embedding is None or len(embedding) == 0:
            # Fallback embedding
            embedding = np.random.randn(80)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        else:
            # Ensure embedding is 1D numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.astype(np.float32)
                if embedding.ndim > 1:
                    embedding = embedding.flatten()

        speaker = SpeakerEmbedding(
            speaker_id=speaker_id,
            embedding=embedding,
            turn_count=1,
            confidence_avg=0.5
        )

        self._speakers[speaker_id] = speaker
        logger.info(f"✨ Registered new speaker {speaker_id}")

        return speaker_id

    def _update_embedding(self, speaker_id: str, new_embedding: np.ndarray, confidence: float):
        """Update a speaker's embedding with running average."""
        if speaker_id not in self._speakers:
            return

        speaker = self._speakers[speaker_id]
        speaker.turn_count += 1

        # Update embedding with exponential moving average (EMA)
        # Newer embeddings get more weight as we see more of this speaker
        alpha = min(0.3, 1.0 / speaker.turn_count)
        speaker.embedding = (1.0 - alpha) * speaker.embedding + alpha * new_embedding

        # Normalize
        speaker.embedding = speaker.embedding / (np.linalg.norm(speaker.embedding) + 1e-9)

        # Update confidence
        speaker.confidence_avg = (speaker.confidence_avg * (speaker.turn_count - 1) + confidence) / speaker.turn_count

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            if a.shape != b.shape:
                return -1.0

            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return -1.0

            similarity = float(np.dot(a, b) / (norm_a * norm_b))
            # Clamp to [-1, 1] range
            return np.clip(similarity, -1.0, 1.0)

        except Exception as e:
            logger.debug(f"Cosine similarity computation failed: {e}")
            return -1.0

    # ══════════════════════════════════════════════════════════════════════════════
    # PUBLIC UTILITIES
    # ══════════════════════════════════════════════════════════════════════════════

    def get_speaker_stats(self) -> Dict:
        """Get statistics about registered speakers."""
        stats = {
            "total_speakers": len(self._speakers),
            "speakers": {}
        }

        for spk_id, speaker in self._speakers.items():
            stats["speakers"][spk_id] = {
                "patient_label": speaker.patient_label,
                "turn_count": speaker.turn_count,
                "confidence": round(speaker.confidence_avg, 3),
                "pitch_mean_hz": round(speaker.pitch_mean, 1),
                "energy_mean_db": round(speaker.energy_mean, 1),
                "embedding_dim": len(speaker.embedding) if speaker.embedding is not None else 0
            }

        return stats

    def compare_all_speakers(self) -> Dict[Tuple[str, str], float]:
        """Compare all pairs of registered speakers."""
        similarities = {}

        speakers_list = list(self._speakers.items())
        for i, (sid1, spk1) in enumerate(speakers_list):
            for sid2, spk2 in speakers_list[i + 1:]:
                sim = self._cosine_similarity(spk1.embedding, spk2.embedding)
                similarities[(sid1, sid2)] = sim

        return similarities
