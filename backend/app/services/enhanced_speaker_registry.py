"""
Enhanced speaker registry using advanced speaker verification.

Replaces the old MFCC-based speaker_registry_service with a modern,
production-grade system using neural speaker embeddings for:
- Voice fingerprinting and uniqueness detection
- Patient-specific labeling (Patient1, Patient2, Patient3)
- Simultaneous speech differentiation
- Confidence scoring at all levels
"""

import logging
from typing import List, Optional, Tuple, Dict
import numpy as np
from app.services.speaker_verification_service import SpeakerVerificationService
from app.services.role_detector_service import SpeakerRole

logger = logging.getLogger(__name__)


class EnhancedSpeakerRegistry:
    """
    Advanced speaker registry using neural speaker embeddings.

    Features:
    - Per-session speaker tracking with voice fingerprints
    - Patient labeling (Patient1, Patient2, Patient3)
    - Doctor/Patient role assignment
    - Confidence scoring
    - Anti-flip-flop logic for role consistency
    - Simultaneous speech differentiation
    """

    def __init__(self, settings=None):
        self.settings = settings or {}
        self.verification_service = SpeakerVerificationService(settings)
        self._speaker_roles: Dict[str, Tuple[str, float, bool]] = {}  # speaker_id -> (role, confidence, locked)
        logger.info("✨ Enhanced speaker registry initialized with neural speaker embeddings")

    def reset(self):
        """Clear all speakers and start fresh."""
        self.verification_service.reset()
        self._speaker_roles.clear()
        logger.info("Enhanced speaker registry reset")

    # ══════════════════════════════════════════════════════════════════════════════
    # SPEAKER IDENTIFICATION
    # ══════════════════════════════════════════════════════════════════════════════

    def identify_speaker(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> Tuple[str, Optional[str], float]:
        """
        Identify a speaker and get their patient label.

        Returns:
            (speaker_id, patient_label, confidence)
            - speaker_id: Internal ID (S1, S2, etc.)
            - patient_label: Patient-visible label (Patient1, Patient2, etc.)
            - confidence: 0-1, how confident in this identification
        """
        try:
            # Use speaker verification service to identify
            speaker_id, confidence, patient_label = self.verification_service.identify_speaker(
                audio, sr
            )

            # Assign patient label if not already assigned
            if patient_label is None:
                patient_label = self.verification_service.assign_patient_label(speaker_id)

            return speaker_id, patient_label, confidence

        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            # Fallback
            speaker_id = f"S_unknown_{np.random.randint(0, 10000)}"
            return speaker_id, "Unknown", 0.0

    def verify_speakers(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        sr: int = 16000
    ) -> Tuple[bool, float]:
        """
        Verify if two audio segments are from the same speaker.

        Returns:
            (is_same_speaker, confidence)
        """
        return self.verification_service.verify_speakers(audio1, audio2, sr)

    # ══════════════════════════════════════════════════════════════════════════════
    # ROLE ASSIGNMENT (with anti-flip-flop)
    # ══════════════════════════════════════════════════════════════════════════════

    def assign_role(
        self,
        speaker_id: str,
        content_role: SpeakerRole,
        content_conf: float
    ) -> Tuple[str, float]:
        """
        Assign/maintain a speaker's role with anti-flip-flop logic.

        Once a role is locked (high confidence), it stays unless there's
        very strong contradicting evidence.

        Returns:
            (role_label, confidence)
        """
        # Get current role if exists
        current_role, current_conf, is_locked = self._speaker_roles.get(
            speaker_id, ("Unknown", 0.0, False)
        )

        lock_threshold = getattr(self.settings, "role_lock_confidence", 75.0)

        # If role is locked, keep it unless new evidence is overwhelming
        if is_locked:
            if (content_role != SpeakerRole.UNKNOWN and
                content_role.value != current_role and
                content_conf >= 90.0):
                # Override locked role only with 90%+ confidence
                self._speaker_roles[speaker_id] = (content_role.value, content_conf, True)
                return content_role.value, content_conf
            else:
                return current_role, current_conf

        # Role not locked - update if we have better evidence
        if content_role != SpeakerRole.UNKNOWN and content_conf > current_conf:
            new_locked = content_conf >= lock_threshold
            self._speaker_roles[speaker_id] = (content_role.value, content_conf, new_locked)

            if new_locked:
                logger.info(f"🔒 Role locked for {speaker_id}: {content_role.value} ({content_conf:.0f}%)")

            return content_role.value, content_conf

        return current_role, current_conf

    # ══════════════════════════════════════════════════════════════════════════════
    # INFORMATION RETRIEVAL
    # ══════════════════════════════════════════════════════════════════════════════

    def get_speaker_info(self, speaker_id: str) -> Dict:
        """Get detailed information about a speaker."""
        patient_label = self.verification_service.get_patient_label(speaker_id)
        role, conf, locked = self._speaker_roles.get(speaker_id, ("Unknown", 0.0, False))

        return {
            "speaker_id": speaker_id,
            "patient_label": patient_label,
            "role": role,
            "role_confidence": round(conf, 2),
            "role_locked": locked
        }

    def get_all_speakers(self) -> Dict:
        """Get information about all speakers in this session."""
        speakers = {}
        for spk_id in self.verification_service._speakers.keys():
            speakers[spk_id] = self.get_speaker_info(spk_id)
        return speakers

    def known_roles(self) -> Dict[str, str]:
        """Get speaker_id -> role mapping."""
        result = {}
        for spk_id in self.verification_service._speakers.keys():
            role, _, _ = self._speaker_roles.get(spk_id, ("Unknown", 0.0, False))
            result[spk_id] = role
        return result

    def known_patients(self) -> Dict[str, str]:
        """Get speaker_id -> patient_label mapping."""
        result = {}
        for spk_id in self.verification_service._speakers.keys():
            patient_label = self.verification_service.get_patient_label(spk_id)
            if patient_label:
                result[spk_id] = patient_label
        return result

    # ══════════════════════════════════════════════════════════════════════════════
    # STATISTICS & DEBUGGING
    # ══════════════════════════════════════════════════════════════════════════════

    def get_session_stats(self) -> Dict:
        """Get comprehensive session statistics."""
        all_speakers = self.verification_service.get_speaker_stats()
        speaker_details = {}

        for spk_id, base_stats in all_speakers.get("speakers", {}).items():
            role, conf, locked = self._speaker_roles.get(spk_id, ("Unknown", 0.0, False))
            speaker_details[spk_id] = {
                **base_stats,
                "role": role,
                "role_confidence": round(conf, 2),
                "role_locked": locked
            }

        return {
            "total_speakers": all_speakers["total_speakers"],
            "speakers": speaker_details,
            "speaker_similarities": self.verification_service.compare_all_speakers()
        }

    def compare_speakers(self, speaker_id1: str, speaker_id2: str) -> Optional[float]:
        """Get similarity score between two speakers."""
        spk1 = self.verification_service._speakers.get(speaker_id1)
        spk2 = self.verification_service._speakers.get(speaker_id2)

        if spk1 is None or spk2 is None:
            return None

        return self.verification_service._cosine_similarity(spk1.embedding, spk2.embedding)

    # ══════════════════════════════════════════════════════════════════════════════
    # PATIENT LABEL MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════════

    def assign_custom_patient_label(self, speaker_id: str, label: str):
        """Manually assign a custom patient label to a speaker."""
        if speaker_id in self.verification_service._speakers:
            self.verification_service._speakers[speaker_id].patient_label = label
            logger.info(f"Assigned patient label '{label}' to {speaker_id}")

    def get_patient_count(self) -> int:
        """Get number of unique patients identified."""
        patients = set()
        for spk in self.verification_service._speakers.values():
            if spk.patient_label:
                patients.add(spk.patient_label)
        return len(patients)
