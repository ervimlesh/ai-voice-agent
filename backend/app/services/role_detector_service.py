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
            "worse", "concerning", "worried", "anxious",
            "sick", "ill", "unwell", "bad", "terrible", "horrible",
            # Time expressions (patient-typical)
            "since", "since last", "for days", "for weeks", "started",
            # Descriptions of experience
            "experiencing", "having", "getting", "feeling",
            "i have", "i'm having", "i've been", "it's been",
            # Self-description of inability / desire (patient asking for help)
            "i am not", "i'm not", "i can't", "i cannot", "i don't",
            "i couldn't", "i didn't", "i want", "i need", "i want to",
            "i need to", "help me", "i'm suffering",
        }

        # Keywords and patterns for doctor speech.
        # Intentionally avoiding generic phrases like "tell me" or "how are you"
        # because patients say those just as often ("tell me how to grow",
        # "how are you doing this test").
        self.doctor_keywords = {
            # Medical questions specific to clinical interview
            "when did", "how long have", "does it", "do you have",
            "describe the",
            # Medical terminology
            "diagnosis", "prescribe", "medication", "treatment", "therapy",
            "examination", "physical exam", "vitals", "blood pressure",
            "infection", "bacterial", "viral",
            "based on", "seems like", "appears to be",
            # Doctor actions / directives
            "i'll prescribe", "i need to examine", "we should run",
            "you should take", "take this", "rest and hydrate",
            "antibiotics", "monitor your", "follow up",
            # Professional language
            "the patient", "your condition", "referring you", "consult",
            "clinical", "healthcare",
        }

        # Patient-typical question patterns (asking for help / advice)
        self.patient_question_patterns = [
            r"what.*do.*i",          # "what do I do"
            r"what.*should.*i",      # "what should i do"
            r"should.*i",            # "should I"
            r"will.*i",              # "will I"
            r"can.*i",               # "can I"
            r"is.*normal",           # "is this normal"
            r"how.*to.*(treat|cure|fix|stop|avoid|grow|gain|lose|reduce|prevent)",
            r"tell.*me.*how",        # "tell me how to grow"
            r"how.*can.*i",          # "how can i"
            r"why.*am.*i",           # "why am i"
            r"why.*do.*i",           # "why do i"
            r"what.*medication",     # "what medication"
            r"i.*want.*to",          # "i want to grow"
            r"i.*need.*to",          # "i need to know"
            r"i.*am.*not",           # "i am not growing"
            r"i'm.*not",             # "i'm not feeling well"
            r"i.*can't",             # "i can't sleep"
            r"i.*don't",             # "i don't feel"
        ]

        # Doctor-typical question patterns (clinical interview)
        self.doctor_question_patterns = [
            r"when.*did.*start",     # "when did it start"
            r"how.*long.*had",       # "how long have you had"
            r"how.*long.*been",      # "how long has it been"
            r"where.*hurt",          # "where does it hurt"
            r"does.*pain",           # "does it cause pain"
            r"any.*other.*symptom",  # "any other symptoms"
            r"have you tried",       # "have you tried"
            r"are you taking",       # "are you taking medication"
            r"allergic",             # "any allergies"
            r"medical history",      # "medical history"
            r"family history",       # "family history"
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
