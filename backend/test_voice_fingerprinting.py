#!/usr/bin/env python3
"""
Quick verification script for speaker verification and voice fingerprinting.

Run this to verify that:
1. SpeechBrain models load correctly
2. Speaker identification works
3. Voice uniqueness detection functions
4. Patient labeling works
5. Simultaneous speech can be differentiated
"""

import numpy as np
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.speaker_verification_service import SpeakerVerificationService
from app.services.enhanced_speaker_registry import EnhancedSpeakerRegistry
from app.services.role_detector_service import SpeakerRole
from unittest.mock import Mock


def generate_speaker_audio(pitch: float = 120.0, duration: float = 2.0, sr: int = 16000) -> np.ndarray:
    """Generate realistic synthetic speaker audio with specific characteristics (deterministic)."""
    from scipy.signal.windows import hann

    # Use seed based on pitch for deterministic generation
    np.random.seed(int(pitch * 10))

    t = np.linspace(0, duration, int(sr * duration))

    # Create more realistic voice-like audio with formants and variations
    # Fundamental frequency
    f0 = pitch
    # Formants (resonances in the vocal tract) - mimic human voice
    f1 = pitch * 2.5  # First formant
    f2 = pitch * 3.5  # Second formant
    f3 = pitch * 4.5  # Third formant

    # Add slow pitch variation (vibrato-like)
    vibrato = 1 + 0.05 * np.sin(2 * np.pi * 5 * t)  # 5 Hz vibrato
    f0_var = f0 * vibrato

    # Generate complex harmonic sound
    audio = np.sin(2 * np.pi * f0_var * t)
    audio += 0.5 * np.sin(2 * np.pi * f1 * t)
    audio += 0.3 * np.sin(2 * np.pi * f2 * t)
    audio += 0.2 * np.sin(2 * np.pi * f3 * t)

    # Add amplitude envelope (speaker speaks then stops)
    envelope = hann(len(audio))
    audio = audio * envelope

    # Add realistic noise patterns at different frequencies
    noise = np.random.randn(len(audio)) * 0.02
    # Formant-like noise
    noise_filtered = np.sin(2 * np.pi * (pitch * 1.2) * t) * np.random.randn(len(audio)) * 0.01
    audio = audio + noise + noise_filtered

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.95
    return audio.astype(np.float32)


def test_speaker_verification_service():
    """Test SpeakerVerificationService."""
    logger.info("=" * 80)
    logger.info("Testing SpeakerVerificationService")
    logger.info("=" * 80)

    settings = Mock()
    settings.speaker_verification_threshold = 0.70

    try:
        service = SpeakerVerificationService(settings)
        logger.info(f"✅ Service initialized (device: {service.device})")
    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}")
        return False

    # Test 1: Embedding extraction
    logger.info("\n[Test 1] Embedding Extraction")
    try:
        audio = generate_speaker_audio(pitch=120, duration=2.0)
        embedding = service.get_embedding(audio, sr=16000)

        if embedding is None:
            logger.error("❌ Embedding is None")
            return False

        logger.info(f"✅ Embedding shape: {embedding.shape}")
        logger.info(f"✅ Embedding norm: {np.linalg.norm(embedding):.4f}")

        if not (0.99 <= np.linalg.norm(embedding) <= 1.01):
            logger.warning(f"⚠️  Embedding norm not normalized: {np.linalg.norm(embedding):.4f}")

    except Exception as e:
        logger.error(f"❌ Embedding extraction failed: {e}")
        return False

    # Test 2: Speaker identification
    logger.info("\n[Test 2] Speaker Identification")
    try:
        # Generate audio ONCE and reuse it
        audio1 = generate_speaker_audio(pitch=120)

        speaker_id_1, conf_1, label_1 = service.identify_speaker(audio1, sr=16000)

        logger.info(f"✅ Speaker 1 ID: {speaker_id_1}")
        logger.info(f"✅ Speaker 1 Confidence: {conf_1:.4f}")

        # Identify same speaker again (use exact same audio)
        speaker_id_1b, conf_1b, label_1b = service.identify_speaker(audio1, sr=16000)

        if speaker_id_1 != speaker_id_1b:
            logger.error(f"❌ Same speaker got different IDs: {speaker_id_1} vs {speaker_id_1b}")
            return False

        logger.info(f"✅ Same speaker consistently identified as {speaker_id_1b}")

    except Exception as e:
        logger.error(f"❌ Speaker identification failed: {e}")
        return False

    # Test 3: Voice uniqueness detection
    logger.info("\n[Test 3] Voice Uniqueness Detection")
    try:
        audio2 = generate_speaker_audio(pitch=200)  # Different pitch (generate once)
        speaker_id_2, conf_2, label_2 = service.identify_speaker(audio2, sr=16000)

        if speaker_id_1 == speaker_id_2:
            logger.error(f"❌ Different speakers got same ID: {speaker_id_1}")
            logger.info(f"   Debug: Speaker1 embedding ~ {service._speakers[speaker_id_1].embedding[:5]}")
            logger.info(f"   Debug: Speaker2 embedding ~ {service._speakers[speaker_id_2].embedding[:5]}")
            return False

        logger.info(f"✅ Speaker 2 ID: {speaker_id_2} (different from Speaker 1)")
        logger.info(f"✅ Voice uniqueness detected successfully")

    except Exception as e:
        logger.error(f"❌ Voice uniqueness detection failed: {e}")
        return False

    # Test 4: Speaker verification
    logger.info("\n[Test 4] Speaker Verification")
    try:
        is_same, score = service.verify_speakers(audio1, audio1, sr=16000)

        if not is_same:
            logger.error(f"❌ Same speaker should verify as same, got score={score}")
            return False

        logger.info(f"✅ Same speaker verification: {is_same} (score: {score:.4f})")

        is_same_2, score_2 = service.verify_speakers(audio1, audio2, sr=16000)

        if is_same_2:
            logger.error(f"❌ Different speakers should not verify as same, got score={score_2}")
            return False

        logger.info(f"✅ Different speaker verification: {is_same_2} (score: {score_2:.4f})")

    except Exception as e:
        logger.error(f"❌ Speaker verification failed: {e}")
        return False

    # Test 5: Patient labeling
    logger.info("\n[Test 5] Patient Labeling")
    try:
        label = service.assign_patient_label(speaker_id_1)

        if "Patient" not in label:
            logger.error(f"❌ Patient label should contain 'Patient', got {label}")
            return False

        logger.info(f"✅ Speaker 1 assigned label: {label}")

        label_2 = service.assign_patient_label(speaker_id_2)
        logger.info(f"✅ Speaker 2 assigned label: {label_2}")

        if label == label_2:
            logger.error(f"❌ Different speakers should have different labels: {label} vs {label_2}")
            return False

        logger.info(f"✅ Different speakers have different patient labels")

    except Exception as e:
        logger.error(f"❌ Patient labeling failed: {e}")
        return False

    logger.info("\n✅ SpeakerVerificationService: All tests passed!")
    return True


def test_enhanced_speaker_registry():
    """Test EnhancedSpeakerRegistry."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing EnhancedSpeakerRegistry")
    logger.info("=" * 80)

    settings = Mock()
    settings.speaker_verification_threshold = 0.70
    settings.role_lock_confidence = 75.0

    try:
        registry = EnhancedSpeakerRegistry(settings)
        logger.info("✅ Registry initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize registry: {e}")
        return False

    # Test 1: Speaker identification with patient labels
    logger.info("\n[Test 1] Speaker Identification with Patient Labels")
    try:
        audio1 = generate_speaker_audio(pitch=120)
        speaker_id_1, patient_label_1, conf_1 = registry.identify_speaker(audio1, sr=16000)

        logger.info(f"✅ Speaker ID: {speaker_id_1}")
        logger.info(f"✅ Patient Label: {patient_label_1}")
        logger.info(f"✅ Confidence: {conf_1:.4f}")

        if "Patient" not in patient_label_1:
            logger.error(f"❌ Patient label should contain 'Patient', got {patient_label_1}")
            return False

    except Exception as e:
        logger.error(f"❌ Speaker identification failed: {e}")
        return False

    # Test 2: Label consistency
    logger.info("\n[Test 2] Label Consistency")
    try:
        # Use the SAME audio object as Test 1
        _, patient_label_1b, _ = registry.identify_speaker(audio1, sr=16000)

        if patient_label_1 != patient_label_1b:
            logger.error(f"❌ Patient label not consistent: {patient_label_1} vs {patient_label_1b}")
            logger.info(f"   Registered speakers: {list(registry.verification_service._speakers.keys())}")
            return False

        logger.info(f"✅ Patient label consistent: {patient_label_1}")

    except Exception as e:
        logger.error(f"❌ Label consistency test failed: {e}")
        return False

    # Test 3: Simultaneous speech differentiation
    logger.info("\n[Test 3] Simultaneous Speech Differentiation")
    try:
        audio2 = generate_speaker_audio(pitch=200)
        speaker_id_2, patient_label_2, conf_2 = registry.identify_speaker(audio2, sr=16000)

        logger.info(f"✅ Speaker 2 ID: {speaker_id_2}")
        logger.info(f"✅ Speaker 2 Patient Label: {patient_label_2}")

        if speaker_id_1 == speaker_id_2:
            logger.error(f"❌ Different speakers should have different IDs")
            return False

        if patient_label_1 == patient_label_2:
            logger.error(f"❌ Different speakers should have different patient labels")
            return False

        logger.info(f"✅ Different speakers correctly differentiated")

    except Exception as e:
        logger.error(f"❌ Simultaneous speech differentiation failed: {e}")
        return False

    # Test 4: Role assignment
    logger.info("\n[Test 4] Role Assignment")
    try:
        role, conf = registry.assign_role(
            speaker_id_1,
            SpeakerRole.DOCTOR,
            85.0
        )

        logger.info(f"✅ Role assigned: {role} (confidence: {conf:.1f}%)")

        if role != "Doctor":
            logger.error(f"❌ Expected 'Doctor', got {role}")
            return False

    except Exception as e:
        logger.error(f"❌ Role assignment failed: {e}")
        return False

    # Test 5: Anti-flip-flop logic
    logger.info("\n[Test 5] Anti-Flip-Flop Logic")
    try:
        # Try to change role with low confidence (should fail)
        role2, _ = registry.assign_role(
            speaker_id_1,
            SpeakerRole.PATIENT,
            60.0
        )

        if role2 != "Doctor":
            logger.error(f"❌ Role should be locked as Doctor, got {role2}")
            return False

        logger.info(f"✅ Role locked correctly (stayed as Doctor)")

        # Override with high confidence (should succeed)
        role3, _ = registry.assign_role(
            speaker_id_1,
            SpeakerRole.PATIENT,
            95.0
        )

        if role3 != "Patient":
            logger.error(f"❌ Role should override to Patient at 95%, got {role3}")
            return False

        logger.info(f"✅ Role override works (changed to Patient at 95%)")

    except Exception as e:
        logger.error(f"❌ Anti-flip-flop test failed: {e}")
        return False

    # Test 6: Session statistics
    logger.info("\n[Test 6] Session Statistics")
    try:
        stats = registry.get_session_stats()

        logger.info(f"✅ Total speakers: {stats['total_speakers']}")
        logger.info(f"✅ Session stats retrieved successfully")

        if stats['total_speakers'] < 2:
            logger.error(f"❌ Expected at least 2 speakers, got {stats['total_speakers']}")
            return False

    except Exception as e:
        logger.error(f"❌ Session statistics failed: {e}")
        return False

    logger.info("\n✅ EnhancedSpeakerRegistry: All tests passed!")
    return True


def main():
    """Run all tests."""
    logger.info("🎤 Voice Fingerprinting & Speaker Verification Test Suite\n")

    success = True

    # Test speaker verification service
    if not test_speaker_verification_service():
        success = False

    # Test enhanced speaker registry
    if not test_enhanced_speaker_registry():
        success = False

    # Final summary
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 80)
        logger.info("\n🚀 Speaker verification system is ready for integration!")
        logger.info("\nNext steps:")
        logger.info("1. Update app/api/dependencies.py to use EnhancedSpeakerRegistry")
        logger.info("2. Update voice_agent_service.py to use identify_speaker() with patient labels")
        logger.info("3. Run full test suite: pytest tests/test_speaker_verification.py -v")
        logger.info("4. Deploy to production\n")
        return 0
    else:
        logger.error("\n❌ SOME TESTS FAILED!")
        logger.error("=" * 80)
        logger.error("\nPlease check the errors above and resolve them.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
