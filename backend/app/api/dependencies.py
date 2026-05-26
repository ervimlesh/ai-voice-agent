from functools import lru_cache

from app.core.config import get_settings
from app.services.medical_rag_service import MedicalRAGService
from app.services.ollama_service import OllamaService
from app.services.voice_agent_service import VoiceAgentService
from app.services.whisper_service import WhisperService
from app.services.speaker_detector_service import SpeakerDetector
from app.services.diarization_service import DiarizationService
from app.services.separation_service import SeparationService
from app.services.voice_embedding_service import VoiceEmbeddingService


@lru_cache
def get_whisper_service() -> WhisperService:
    return WhisperService(get_settings())


@lru_cache
def get_ollama_service() -> OllamaService:
    return OllamaService(get_settings())


@lru_cache
def get_rag_service() -> MedicalRAGService:
    return MedicalRAGService()


@lru_cache
def get_speaker_detector() -> SpeakerDetector:
    return SpeakerDetector()


@lru_cache
def get_diarization_service() -> DiarizationService:
    """Singleton diarizer — the heavy pyannote pipeline loads once here."""
    return DiarizationService(get_settings(), get_speaker_detector())


@lru_cache
def get_separation_service() -> SeparationService:
    """Singleton separator — the heavy Sepformer model loads once here."""
    return SeparationService(get_settings())


@lru_cache
def get_voice_embedding_service() -> VoiceEmbeddingService:
    """Singleton ECAPA-TDNN voice encoder — neural speaker fingerprints."""
    return VoiceEmbeddingService(get_settings())


def get_voice_agent_service() -> VoiceAgentService:
    return VoiceAgentService(get_whisper_service(), get_ollama_service())
