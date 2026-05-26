import re
from typing import Tuple
from enum import Enum


class SpeakerRole(Enum):
    DOCTOR = "Doctor"
    PATIENT = "Patient"
    UNKNOWN = "Unknown"


class RoleDetectorService:
    """
    Detects whether a speaker is a Doctor or Patient based on speech content and patterns.
    Uses heuristics to analyze language patterns typical of medical professionals vs patients.
    """

    def __init__(self):
        # Keywords and patterns for patient speech
        self.patient_keywords = {
            # Symptom descriptions
            "pain", "ache", "hurt", "fever", "cough", "sneeze", "cold", "flu",
            "headache", "nausea", "dizzy", "dizziness", "tired", "fatigue",
            "weak", "weakness", "sore", "rash", "itch", "itching", "bleeding",
            "discharge", "swelling", "inflammation", "burn", "burning",
            "symptom", "symptoms", "suffering", "trouble", "problem", "issue",
            "worse", "better", "worse", "concerning", "worried", "anxious",
            "sick", "ill", "unwell", "bad", "terrible", "horrible",
            # Time expressions (patient-typical)
            "since", "since last", "for days", "for weeks", "started",
            # Descriptions of experience
            "experiencing", "having", "getting", "feeling",
            "i have", "i'm having", "i've been", "it's been",
        }

        # Keywords and patterns for doctor speech
        self.doctor_keywords = {
            # Medical questions
            "when did", "how long", "does it", "do you have",
            "any", "have you", "describe", "tell me", "how are you",
            # Medical terminology
            "diagnosis", "prescribe", "medication", "treatment", "therapy",
            "examination", "physical", "vitals", "blood pressure",
            "infection", "inflammation", "bacterial", "viral",
            "based on", "seems like", "appears to be", "likely",
            # Doctor actions
            "let me", "i'll", "i need to", "we should", "you should",
            "take", "rest", "hydrate", "medication", "antibiotics",
            "monitor", "follow up", "come back", "see you",
            # Professional language
            "patient", "condition", "case", "referring", "consult",
            "clinical", "medical", "health", "healthcare",
        }

        # Patient-typical question patterns
        self.patient_question_patterns = [
            r"what.*do.*i",  # "what do I do"
            r"should.*i",    # "should I"
            r"will.*i",      # "will I"
            r"can.*i",       # "can I"
            r"is.*normal",   # "is this normal"
            r"how.*treat",   # "how to treat"
            r"what.*medication",  # "what medication"
        ]

        # Doctor-typical question patterns
        self.doctor_question_patterns = [
            r"when.*start",  # "when did it start"
            r"how.*long",    # "how long has it"
            r"where.*hurt",  # "where does it hurt"
            r"does.*pain",   # "does it cause pain"
            r"any.*other",   # "any other symptoms"
            r"have you tried",  # "have you tried"
            r"take.*medication",  # "are you taking medication"
            r"allergic",     # "any allergies"
            r"history",      # "medical history"
        ]

    def detect_role(self, text: str) -> Tuple[SpeakerRole, float]:
        """
        Detect if speaker is Doctor or Patient based on text content.

        Returns:
            (SpeakerRole, confidence_score) where confidence is 0-100
        """
        if not text or len(text.strip()) < 3:
            return SpeakerRole.UNKNOWN, 0.0

        text_lower = text.lower()
        words = set(text_lower.split())

        # Calculate keyword matches
        patient_score = sum(1 for keyword in self.patient_keywords if keyword in text_lower)
        doctor_score = sum(1 for keyword in self.doctor_keywords if keyword in text_lower)

        # Check question patterns
        for pattern in self.patient_question_patterns:
            if re.search(pattern, text_lower):
                patient_score += 2

        for pattern in self.doctor_question_patterns:
            if re.search(pattern, text_lower):
                doctor_score += 2

        # Analyze pronouns and perspective
        if "i " in text_lower or "i'm" in text_lower or "i've" in text_lower:
            if "my " in text_lower or "my " in text_lower:
                patient_score += 1  # "my symptoms", "my pain"

        if "you " in text_lower or "your " in text_lower:
            doctor_score += 1  # Addressing the patient

        # Calculate confidence
        total_score = patient_score + doctor_score
        if total_score == 0:
            return SpeakerRole.UNKNOWN, 0.0

        patient_confidence = (patient_score / total_score) * 100
        doctor_confidence = (doctor_score / total_score) * 100

        # Determine role
        if patient_score > doctor_score:
            return SpeakerRole.PATIENT, patient_confidence
        elif doctor_score > patient_score:
            return SpeakerRole.DOCTOR, doctor_confidence
        else:
            return SpeakerRole.UNKNOWN, 50.0

    def should_switch_roles(self, previous_role: SpeakerRole, current_role: SpeakerRole) -> bool:
        """
        Determine if speaker has switched roles based on detection.

        Includes anti-flicker logic to avoid rapid role switches.
        """
        if previous_role == SpeakerRole.UNKNOWN or current_role == SpeakerRole.UNKNOWN:
            return False

        return previous_role != current_role

    def merge_role_detections(
        self,
        voice_role: SpeakerRole,
        voice_confidence: float,
        content_role: SpeakerRole,
        content_confidence: float,
    ) -> Tuple[SpeakerRole, float]:
        """
        Merge voice-based and content-based role detection.

        Args:
            voice_role: Role from voice analysis (Doctor/Patient)
            voice_confidence: Confidence 0-100 from voice
            content_role: Role from content analysis (Doctor/Patient)
            content_confidence: Confidence 0-100 from content

        Returns:
            (merged_role, merged_confidence)
        """
        if content_role == SpeakerRole.UNKNOWN:
            return voice_role, voice_confidence

        if voice_role == SpeakerRole.UNKNOWN:
            return content_role, content_confidence

        # If both agree, boost confidence
        if voice_role == content_role:
            merged_confidence = min(100, (voice_confidence + content_confidence) / 2 * 1.2)
            return voice_role, merged_confidence

        # If they disagree, weight by confidence
        if voice_confidence > content_confidence:
            merged_confidence = voice_confidence * 0.7 + content_confidence * 0.3
            return voice_role, merged_confidence
        else:
            merged_confidence = content_confidence * 0.7 + voice_confidence * 0.3
            return content_role, merged_confidence

    def get_role_display(self, role: SpeakerRole, confidence: float) -> str:
        """Get human-readable role display string."""
        if role == SpeakerRole.UNKNOWN:
            return "Unknown"
        return f"{role.value} ({confidence:.0f}%)"
