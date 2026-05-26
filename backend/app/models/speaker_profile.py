from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SpeakerProfile:
    """Stores voice characteristics for speaker identification"""
    pitch: float
    spectral_centroid: float
    mfcc: List[float]
    zero_crossing_rate: float
    rms_energy: float
    speech_rate: float = 0.0  # words per minute
    # Dense L2-normalized embedding used as the primary identity signal.
    # Tier A: pyannote/SpeechBrain speaker embedding; Tier B: stacked MFCC stats.
    embedding: Optional[List[float]] = None

    def to_dict(self) -> dict:
        """Convert profile to dictionary for storage"""
        return {
            "pitch": self.pitch,
            "spectral_centroid": self.spectral_centroid,
            "mfcc": self.mfcc,
            "zero_crossing_rate": self.zero_crossing_rate,
            "rms_energy": self.rms_energy,
            "speech_rate": self.speech_rate,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SpeakerProfile':
        """Create profile from dictionary"""
        return cls(
            pitch=data["pitch"],
            spectral_centroid=data["spectral_centroid"],
            mfcc=data["mfcc"],
            zero_crossing_rate=data["zero_crossing_rate"],
            rms_energy=data["rms_energy"],
            speech_rate=data.get("speech_rate", 0.0),
            embedding=data.get("embedding"),
        )


@dataclass
class RegisteredSpeaker:
    """A persistent identity within one conversation (S1, S2, S3, ...).

    The embedding is a running average so the identity stays stable while
    adapting slightly to new utterances from the same person.
    """
    speaker_id: str                 # "S1", "S2", ...
    embedding: np.ndarray           # running-average identity vector
    role: str = "Unknown"           # Doctor | Patient | Relative | Unknown
    role_confidence: float = 0.0
    role_locked: bool = False       # once confidently set, resist flip-flop
    turn_count: int = 0
