from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="AI Voice Agent API", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ORIGINS",
    )
    whisper_model: str = Field(default="medium", alias="WHISPER_MODEL")
    whisper_task: str = Field(default="transcribe", alias="WHISPER_TASK")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")
    # Force a specific language ("en", "hi", etc.) or leave None for auto-detect.
    # Forcing language dramatically reduces script-misdetection garbage when the
    # speaker's primary language is known.
    whisper_language: str | None = Field(default=None, alias="WHISPER_LANGUAGE")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    max_history_messages: int = Field(default=10, alias="MAX_HISTORY_MESSAGES")

    # ── Diarization / multi-speaker settings ──
    diarization_enabled: bool = Field(default=True, alias="DIARIZATION_ENABLED")
    # "auto" picks pyannote when available, otherwise the lightweight clustering tier.
    diarization_backend: str = Field(default="auto", alias="DIARIZATION_BACKEND")
    # "auto" picks cuda when a GPU is present, otherwise cpu.
    diarization_device: str = Field(default="auto", alias="DIARIZATION_DEVICE")
    # HuggingFace token required to download pyannote models (Tier A only).
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    # Cosine-similarity threshold (0-1) to treat a voice as an already-known speaker.
    speaker_match_threshold: float = Field(default=0.78, alias="SPEAKER_MATCH_THRESHOLD")
    # Sub-segments shorter than this (seconds) are ignored as too short to attribute.
    min_turn_duration_s: float = Field(default=0.4, alias="MIN_TURN_DURATION_S")
    # Above this content-confidence (0-100) a speaker's role is locked to stop flip-flop.
    role_lock_confidence: float = Field(default=75.0, alias="ROLE_LOCK_CONFIDENCE")

    # ── Neural voice embeddings (ECAPA-TDNN speaker fingerprints) ──
    # When enabled and the model loads, per-turn speaker identity uses neural
    # embeddings instead of MFCC stats — much more accurate, incl. overlap.
    voice_embedding_enabled: bool = Field(default=True, alias="VOICE_EMBEDDING_ENABLED")
    voice_embedding_model: str = Field(
        default="speechbrain/spkrec-ecapa-voxceleb", alias="VOICE_EMBEDDING_MODEL"
    )
    # "auto" picks cuda when a GPU is present, otherwise cpu.
    voice_embedding_device: str = Field(default="auto", alias="VOICE_EMBEDDING_DEVICE")
    # Cosine-similarity threshold tuned for ECAPA embeddings (lower than MFCC's).
    ecapa_match_threshold: float = Field(default=0.50, alias="ECAPA_MATCH_THRESHOLD")

    # ── Source separation (overlap un-mixing, Tier A) ──
    separation_enabled: bool = Field(default=True, alias="SEPARATION_ENABLED")
    # "auto" un-mixes with Sepformer only when it loads (GPU-class); else flags overlap.
    separation_backend: str = Field(default="auto", alias="SEPARATION_BACKEND")
    # Fraction of a segment that must be flagged as overlap before we attempt un-mixing.
    overlap_min_ratio: float = Field(default=0.15, alias="OVERLAP_MIN_RATIO")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def allowed_origins(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
