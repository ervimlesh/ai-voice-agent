"""
Test suite for speaker verification and voice fingerprinting.

Tests:
- Voice uniqueness detection
- Speaker identification
- Speaker verification (same person detection)
- Patient labeling (Patient1, Patient2, Patient3)
- Simultaneous speech differentiation
- Confidence scoring
"""

import pytest
import numpy as np
import librosa
from app.services.speaker_verification_service import SpeakerVerificationService
from app.services.enhanced_speaker_registry import EnhancedSpeakerRegistry
from app.services.role_detector_service import SpeakerRole
from unittest.mock import Mock


class TestSpeakerVerificationService:
    """Test speaker verification and embedding extraction."""

    @pytest.fixture
    def service(self):
        """Create a speaker verification service."""
        settings = Mock()
        settings.speaker_verification_threshold = 0.70
        return SpeakerVerificationService(settings)

    @pytest.fixture
    def synthetic_speaker_audio(self):
        """Generate synthetic speaker audio with consistent characteristics."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create audio with specific pitch and timbre characteristics
        pitch = 120  # Hz - consistent pitch
        audio = np.sin(2 * np.pi * pitch * t)  # Base tone
        audio += 0.3 * np.sin(2 * np.pi * pitch * 2 * t)  # Harmonic
        audio += 0.2 * np.sin(2 * np.pi * pitch * 3 * t)  # Harmonic

        # Add slight variations to make it realistic
        noise = np.random.randn(len(audio)) * 0.01
        audio = audio + noise

        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.95

        return audio.astype(np.float32), sr

    @pytest.fixture
    def different_speaker_audio(self):
        """Generate different speaker audio with different characteristics."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Different pitch and timbre
        pitch = 200  # Hz - much different pitch
        audio = np.sin(2 * np.pi * pitch * t)
        audio += 0.4 * np.sin(2 * np.pi * pitch * 1.5 * t)  # Different harmonics
        audio += 0.15 * np.sin(2 * np.pi * pitch * 2.5 * t)

        noise = np.random.randn(len(audio)) * 0.01
        audio = audio + noise
        audio = audio / np.max(np.abs(audio)) * 0.95

        return audio.astype(np.float32), sr

    # ──────────────────────────────────────────────────────────────────────────────
    # Embedding Extraction Tests
    # ──────────────────────────────────────────────────────────────────────────────

    def test_embedding_extraction_shape(self, service, synthetic_speaker_audio):
        """Test that embeddings have correct shape."""
        audio, sr = synthetic_speaker_audio
        embedding = service.get_embedding(audio, sr)

        assert embedding is not None, "Embedding should not be None"
        assert len(embedding) > 0, "Embedding should have dimensions"
        assert embedding.shape[0] >= 80, "Embedding should be at least 80D (fallback)"

    def test_embedding_normalization(self, service, synthetic_speaker_audio):
        """Test that embeddings are normalized to unit length."""
        audio, sr = synthetic_speaker_audio
        embedding = service.get_embedding(audio, sr)

        if embedding is not None:
            norm = np.linalg.norm(embedding)
            assert 0.99 <= norm <= 1.01, f"Embedding should be normalized, got norm={norm}"

    def test_embedding_consistency(self, service, synthetic_speaker_audio):
        """Test that same speaker produces similar embeddings."""
        audio, sr = synthetic_speaker_audio

        emb1 = service.get_embedding(audio, sr)
        emb2 = service.get_embedding(audio, sr)

        assert emb1 is not None and emb2 is not None, "Embeddings should not be None"

        # Embeddings should be identical or very similar
        similarity = service._cosine_similarity(emb1, emb2)
        assert similarity > 0.95, f"Same audio should produce similar embeddings, got {similarity}"

    def test_different_speakers_different_embeddings(
        self,
        service,
        synthetic_speaker_audio,
        different_speaker_audio
    ):
        """Test that different speakers produce different embeddings."""
        audio1, sr1 = synthetic_speaker_audio
        audio2, sr2 = different_speaker_audio

        emb1 = service.get_embedding(audio1, sr1)
        emb2 = service.get_embedding(audio2, sr2)

        assert emb1 is not None and emb2 is not None, "Embeddings should not be None"

        similarity = service._cosine_similarity(emb1, emb2)
        assert similarity < 0.85, f"Different speakers should have low similarity, got {similarity}"

    def test_embedding_short_audio_handling(self, service):
        """Test embedding extraction with short audio."""
        sr = 16000
        # Very short audio (0.1 seconds)
        short_audio = np.random.randn(1600).astype(np.float32)

        embedding = service.get_embedding(short_audio, sr)
        # Should either return None or a fallback embedding
        assert embedding is None or len(embedding) > 0

    def test_embedding_empty_audio_handling(self, service):
        """Test embedding extraction with empty audio."""
        embedding = service.get_embedding(np.array([]), 16000)
        assert embedding is None, "Empty audio should return None"

    # ──────────────────────────────────────────────────────────────────────────────
    # Speaker Identification Tests
    # ──────────────────────────────────────────────────────────────────────────────

    def test_identify_first_speaker(self, service, synthetic_speaker_audio):
        """Test identifying first speaker (should register as new)."""
        audio, sr = synthetic_speaker_audio

        speaker_id, confidence, patient_label = service.identify_speaker(audio, sr)

        assert speaker_id is not None, "Should return speaker ID"
        assert speaker_id.startswith("S"), "Speaker ID should start with S"
        # First speaker might have lower confidence
        assert 0.0 <= confidence <= 1.0, "Confidence should be 0-1"

    def test_identify_same_speaker_twice(self, service, synthetic_speaker_audio):
        """Test that same speaker is identified consistently."""
        audio, sr = synthetic_speaker_audio

        # Identify first time
        id1, conf1, _ = service.identify_speaker(audio, sr)

        # Identify same speaker again
        id2, conf2, _ = service.identify_speaker(audio, sr)

        # Should match
        assert id1 == id2, f"Same speaker should have same ID: {id1} vs {id2}"
        assert conf2 >= conf1, "Confidence should increase or stay same on repeat"

    def test_identify_different_speakers_separately(
        self,
        service,
        synthetic_speaker_audio,
        different_speaker_audio
    ):
        """Test that different speakers get different IDs."""
        audio1, sr1 = synthetic_speaker_audio
        audio2, sr2 = different_speaker_audio

        id1, _, _ = service.identify_speaker(audio1, sr1)
        id2, _, _ = service.identify_speaker(audio2, sr2)

        assert id1 != id2, f"Different speakers should have different IDs: {id1} vs {id2}"

    def test_identify_empty_audio(self, service):
        """Test identifying empty audio."""
        speaker_id, confidence, patient_label = service.identify_speaker(np.array([]), 16000)

        assert speaker_id is not None, "Should still return speaker ID"
        assert 0.0 <= confidence <= 1.0, "Confidence should be 0-1"

    # ──────────────────────────────────────────────────────────────────────────────
    # Speaker Verification Tests
    # ──────────────────────────────────────────────────────────────────────────────

    def test_verify_same_speaker(self, service, synthetic_speaker_audio):
        """Test verifying same speaker."""
        audio, sr = synthetic_speaker_audio

        is_same, confidence = service.verify_speakers(audio, audio, sr)

        assert is_same is True, "Same audio should verify as same speaker"
        assert confidence > 0.8, f"High confidence expected for same speaker, got {confidence}"

    def test_verify_different_speakers(
        self,
        service,
        synthetic_speaker_audio,
        different_speaker_audio
    ):
        """Test verifying different speakers."""
        audio1, sr1 = synthetic_speaker_audio
        audio2, sr2 = different_speaker_audio

        is_same, confidence = service.verify_speakers(audio1, audio2, sr1)

        assert is_same is False, "Different speakers should not verify as same"
        assert confidence < 0.7, f"Low confidence expected for different speakers, got {confidence}"

    def test_verify_with_empty_audio(self, service, synthetic_speaker_audio):
        """Test verification with empty audio."""
        audio, sr = synthetic_speaker_audio

        is_same, confidence = service.verify_speakers(audio, np.array([]), sr)

        assert is_same is False, "Empty audio should not verify as same"
        assert confidence <= 0.0, "Low confidence for empty audio"

    # ──────────────────────────────────────────────────────────────────────────────
    # Patient Labeling Tests
    # ──────────────────────────────────────────────────────────────────────────────

    def test_patient_label_assignment(self, service, synthetic_speaker_audio):
        """Test assigning patient labels."""
        audio, sr = synthetic_speaker_audio

        speaker_id, _, initial_label = service.identify_speaker(audio, sr)

        # Assign patient label
        label = service.assign_patient_label(speaker_id)

        assert label.startswith("Patient"), f"Patient label should start with 'Patient', got {label}"
        assert label[7:].isdigit(), "Patient label should have number"

    def test_patient_label_consistency(self, service, synthetic_speaker_audio):
        """Test that patient label stays consistent."""
        audio, sr = synthetic_speaker_audio

        speaker_id, _, _ = service.identify_speaker(audio, sr)

        label1 = service.assign_patient_label(speaker_id)
        label2 = service.assign_patient_label(speaker_id)

        assert label1 == label2, f"Patient label should be consistent: {label1} vs {label2}"

    def test_multiple_patient_labels(
        self,
        service,
        synthetic_speaker_audio,
        different_speaker_audio
    ):
        """Test that different speakers get different patient labels."""
        audio1, sr1 = synthetic_speaker_audio
        audio2, sr2 = different_speaker_audio

        id1, _, _ = service.identify_speaker(audio1, sr1)
        label1 = service.assign_patient_label(id1)

        id2, _, _ = service.identify_speaker(audio2, sr2)
        label2 = service.assign_patient_label(id2)

        assert label1 != label2, f"Different speakers should have different labels: {label1} vs {label2}"
        assert "Patient1" in label1 or "Patient2" in label1
        assert "Patient1" in label2 or "Patient2" in label2


class TestEnhancedSpeakerRegistry:
    """Test the enhanced speaker registry with role assignment."""

    @pytest.fixture
    def registry(self):
        """Create an enhanced speaker registry."""
        settings = Mock()
        settings.speaker_verification_threshold = 0.70
        settings.role_lock_confidence = 75.0
        return EnhancedSpeakerRegistry(settings)

    @pytest.fixture
    def synthetic_audio(self):
        """Generate synthetic audio."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        pitch = 120
        audio = np.sin(2 * np.pi * pitch * t)
        audio += 0.3 * np.sin(2 * np.pi * pitch * 2 * t)
        noise = np.random.randn(len(audio)) * 0.01
        audio = audio + noise
        return (audio / np.max(np.abs(audio)) * 0.95).astype(np.float32), sr

    # ──────────────────────────────────────────────────────────────────────────────
    # Speaker Identification with Registry
    # ──────────────────────────────────────────────────────────────────────────────

    def test_identify_with_patient_label(self, registry, synthetic_audio):
        """Test speaker identification returns patient label."""
        audio, sr = synthetic_audio

        speaker_id, patient_label, confidence = registry.identify_speaker(audio, sr)

        assert speaker_id is not None, "Should return speaker ID"
        assert patient_label is not None, "Should return patient label"
        assert "Patient" in patient_label, f"Patient label should contain 'Patient', got {patient_label}"
        assert 0.0 <= confidence <= 1.0, "Confidence should be 0-1"

    def test_patient_label_consistency_in_registry(self, registry, synthetic_audio):
        """Test that patient label stays consistent across identifications."""
        audio, sr = synthetic_audio

        _, label1, _ = registry.identify_speaker(audio, sr)
        _, label2, _ = registry.identify_speaker(audio, sr)

        assert label1 == label2, f"Patient label should be consistent: {label1} vs {label2}"

    # ──────────────────────────────────────────────────────────────────────────────
    # Role Assignment
    # ──────────────────────────────────────────────────────────────────────────────

    def test_assign_role_doctor(self, registry, synthetic_audio):
        """Test assigning doctor role."""
        audio, sr = synthetic_audio
        speaker_id, _, _ = registry.identify_speaker(audio, sr)

        role, confidence = registry.assign_role(speaker_id, SpeakerRole.DOCTOR, 80.0)

        assert role == "Doctor", f"Role should be Doctor, got {role}"
        assert confidence == 80.0, f"Confidence should be 80.0, got {confidence}"

    def test_assign_role_patient(self, registry, synthetic_audio):
        """Test assigning patient role."""
        audio, sr = synthetic_audio
        speaker_id, _, _ = registry.identify_speaker(audio, sr)

        role, confidence = registry.assign_role(speaker_id, SpeakerRole.PATIENT, 85.0)

        assert role == "Patient", f"Role should be Patient, got {role}"

    def test_role_lock_prevents_flip_flop(self, registry, synthetic_audio):
        """Test that role locking prevents flip-flopping."""
        audio, sr = synthetic_audio
        speaker_id, _, _ = registry.identify_speaker(audio, sr)

        # Assign first role (high confidence, should lock)
        role1, _ = registry.assign_role(speaker_id, SpeakerRole.DOCTOR, 80.0)
        assert role1 == "Doctor", "Should be Doctor"

        # Try to assign different role with lower confidence (should not change)
        role2, _ = registry.assign_role(speaker_id, SpeakerRole.PATIENT, 60.0)
        assert role2 == "Doctor", "Role should stay Doctor (locked)"

        # Try to assign different role with very high confidence (should override)
        role3, _ = registry.assign_role(speaker_id, SpeakerRole.PATIENT, 95.0)
        assert role3 == "Patient", "Should override locked role with 95% confidence"

    # ──────────────────────────────────────────────────────────────────────────────
    # Speaker Information Retrieval
    # ──────────────────────────────────────────────────────────────────────────────

    def test_get_speaker_info(self, registry, synthetic_audio):
        """Test retrieving speaker information."""
        audio, sr = synthetic_audio
        speaker_id, _, _ = registry.identify_speaker(audio, sr)

        info = registry.get_speaker_info(speaker_id)

        assert "speaker_id" in info
        assert "patient_label" in info
        assert "role" in info
        assert "role_confidence" in info
        assert info["speaker_id"] == speaker_id

    def test_known_patients(self, registry, synthetic_audio):
        """Test retrieving known patients."""
        audio, sr = synthetic_audio
        speaker_id, _, _ = registry.identify_speaker(audio, sr)

        patients = registry.known_patients()

        assert speaker_id in patients, f"Speaker should be in known_patients"
        assert "Patient" in patients[speaker_id]

    def test_patient_count(self, registry, synthetic_audio):
        """Test counting unique patients."""
        audio, sr = synthetic_audio
        registry.identify_speaker(audio, sr)

        count = registry.get_patient_count()
        assert count >= 1, f"Should have at least 1 patient, got {count}"

    # ──────────────────────────────────────────────────────────────────────────────
    # Session Statistics
    # ──────────────────────────────────────────────────────────────────────────────

    def test_session_stats(self, registry, synthetic_audio):
        """Test retrieving session statistics."""
        audio, sr = synthetic_audio
        registry.identify_speaker(audio, sr)

        stats = registry.get_session_stats()

        assert "total_speakers" in stats
        assert "speakers" in stats
        assert stats["total_speakers"] >= 1

    def test_reset_registry(self, registry, synthetic_audio):
        """Test resetting the registry."""
        audio, sr = synthetic_audio
        registry.identify_speaker(audio, sr)

        initial_count = registry.get_patient_count()
        assert initial_count > 0, "Should have identified speakers"

        registry.reset()

        reset_count = registry.get_patient_count()
        assert reset_count == 0, f"Reset should clear speakers, got {reset_count}"


class TestSimultaneousSpeechDifferentiation:
    """Test differentiating speakers in simultaneous speech scenarios."""

    @pytest.fixture
    def registry(self):
        """Create an enhanced speaker registry."""
        settings = Mock()
        settings.speaker_verification_threshold = 0.70
        settings.role_lock_confidence = 75.0
        return EnhancedSpeakerRegistry(settings)

    @pytest.fixture
    def speaker1_audio(self):
        """Generate Speaker 1 audio."""
        sr = 16000
        t = np.linspace(0, 2.0, int(sr * 2.0))
        pitch = 120
        audio = np.sin(2 * np.pi * pitch * t) + 0.3 * np.sin(2 * np.pi * pitch * 2 * t)
        noise = np.random.randn(len(audio)) * 0.01
        audio = audio + noise
        return (audio / np.max(np.abs(audio)) * 0.95).astype(np.float32), sr

    @pytest.fixture
    def speaker2_audio(self):
        """Generate Speaker 2 audio (different pitch)."""
        sr = 16000
        t = np.linspace(0, 2.0, int(sr * 2.0))
        pitch = 200
        audio = np.sin(2 * np.pi * pitch * t) + 0.4 * np.sin(2 * np.pi * pitch * 1.5 * t)
        noise = np.random.randn(len(audio)) * 0.01
        audio = audio + noise
        return (audio / np.max(np.abs(audio)) * 0.95).astype(np.float32), sr

    def test_simultaneous_speech_different_patients(self, registry, speaker1_audio, speaker2_audio):
        """Test identifying different patients in simultaneous speech."""
        audio1, sr1 = speaker1_audio
        audio2, sr2 = speaker2_audio

        # Identify first speaker
        id1, label1, conf1 = registry.identify_speaker(audio1, sr1)
        assert "Patient" in label1, f"Speaker 1 should be labeled as patient, got {label1}"

        # Identify second speaker (overlapping/simultaneous)
        id2, label2, conf2 = registry.identify_speaker(audio2, sr2)
        assert "Patient" in label2, f"Speaker 2 should be labeled as patient, got {label2}"

        # Should be different patients
        assert id1 != id2, "Different speakers should have different IDs"
        assert label1 != label2, f"Different speakers should have different labels: {label1} vs {label2}"

    def test_simultaneous_identification_consistency(
        self,
        registry,
        speaker1_audio,
        speaker2_audio
    ):
        """Test consistent identification of speakers in simultaneous scenarios."""
        audio1, sr1 = speaker1_audio
        audio2, sr2 = speaker2_audio

        # First identification
        id1_first, label1_first, _ = registry.identify_speaker(audio1, sr1)
        id2_first, label2_first, _ = registry.identify_speaker(audio2, sr2)

        # Re-identify same speakers
        id1_second, label1_second, _ = registry.identify_speaker(audio1, sr1)
        id2_second, label2_second, _ = registry.identify_speaker(audio2, sr2)

        # Should match
        assert id1_first == id1_second, f"Speaker 1 IDs should match: {id1_first} vs {id1_second}"
        assert id2_first == id2_second, f"Speaker 2 IDs should match: {id2_first} vs {id2_second}"
        assert label1_first == label1_second, f"Speaker 1 labels should match"
        assert label2_first == label2_second, f"Speaker 2 labels should match"


class TestVoiceUniquenessDetection:
    """Test voice uniqueness detection (fingerprinting)."""

    @pytest.fixture
    def service(self):
        """Create a speaker verification service."""
        settings = Mock()
        settings.speaker_verification_threshold = 0.70
        return SpeakerVerificationService(settings)

    def test_voice_fingerprint_uniqueness(self, service):
        """Test that each voice has a unique fingerprint."""
        sr = 16000

        # Generate 3 different speakers with distinct characteristics
        speakers_audio = []
        for pitch in [100, 150, 200]:
            t = np.linspace(0, 2.0, int(sr * 2.0))
            audio = np.sin(2 * np.pi * pitch * t) * np.exp(-t / 4)  # Decaying tone
            noise = np.random.randn(len(audio)) * 0.01
            audio = audio + noise
            audio = (audio / np.max(np.abs(audio)) * 0.95).astype(np.float32)
            speakers_audio.append(audio)

        # Extract embeddings
        embeddings = [service.get_embedding(audio, sr) for audio in speakers_audio]

        # All embeddings should exist
        assert all(e is not None for e in embeddings), "All embeddings should be valid"

        # Compare all pairs
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = service._cosine_similarity(embeddings[i], embeddings[j])
                assert sim < 0.75, (
                    f"Different speakers should have low similarity, "
                    f"Speaker {i} vs {j}: {sim}"
                )

    def test_fingerprint_stability_over_time(self, service):
        """Test that voice fingerprints are stable for same speaker."""
        sr = 16000
        pitch = 120

        # Generate same speaker at different times
        embeddings = []
        for offset in range(3):
            t = np.linspace(0, 2.0, int(sr * 2.0))
            # Slight variations but same speaker
            audio = np.sin(2 * np.pi * (pitch + offset * 2) * t)
            audio += 0.3 * np.sin(2 * np.pi * (pitch + offset * 2) * 2 * t)
            noise = np.random.randn(len(audio)) * (0.01 + offset * 0.005)
            audio = audio + noise
            audio = (audio / np.max(np.abs(audio)) * 0.95).astype(np.float32)

            emb = service.get_embedding(audio, sr)
            embeddings.append(emb)

        # All embeddings should be similar
        for i in range(len(embeddings) - 1):
            sim = service._cosine_similarity(embeddings[i], embeddings[i + 1])
            assert sim > 0.60, (
                f"Same speaker should have high stability, "
                f"Embedding {i} vs {i+1}: {sim}"
            )
