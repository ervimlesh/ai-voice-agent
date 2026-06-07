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
    # Cosine-similarity threshold tuned for ECAPA embeddings. Lowered from 0.50
    # → 0.40 for multi-voice setups where the same physical voice arrives at
    # variable volumes/distances (own voice vs phone speaker held near the mic
    # vs ChatGPT through laptop speakers). At 0.50 the embeddings for the same
    # person at different distances drifted below threshold and got registered
    # as S3/S4/S5. 0.40 keeps real voice identity stable while still cleanly
    # separating different people (different-speaker similarity typically <0.3).
    ecapa_match_threshold: float = Field(default=0.40, alias="ECAPA_MATCH_THRESHOLD")

    # ── Source separation (overlap un-mixing, Tier A) ──
    separation_enabled: bool = Field(default=True, alias="SEPARATION_ENABLED")
    # "auto" un-mixes with Sepformer only when it loads (GPU-class); else flags overlap.
    separation_backend: str = Field(default="auto", alias="SEPARATION_BACKEND")
    # Fraction of a segment that must be flagged as overlap before we attempt un-mixing.
    overlap_min_ratio: float = Field(default=0.15, alias="OVERLAP_MIN_RATIO")
    # A separated Sepformer source is kept only if its energy (RMS) is at least
    # this fraction of the loudest source. Sepformer always emits 2 sources even
    # for single-voice audio; the phantom 2nd source is low-energy and, if kept,
    # makes Whisper hallucinate words nobody said. Raise to be stricter (drop
    # more phantoms), lower if a genuine quiet 2nd voice is being discarded.
    separation_stream_rel_floor: float = Field(default=0.30, alias="SEPARATION_STREAM_REL_FLOOR")
    # Minimum duration (seconds) of a time-overlap BETWEEN two diarized turns
    # before we force-un-mix that shared region. Pyannote often emits tiny
    # (<0.3s) boundary overlaps that are just padding, not real talk-over; this
    # ignores those while catching genuine 1-2s talk-over that the per-clip
    # ratio heuristic dilutes below overlap_min_ratio. Lower to catch shorter
    # talk-over, raise to be more conservative.
    overlap_min_region_s: float = Field(default=0.3, alias="OVERLAP_MIN_REGION_S")

    # ── Realtime WebSocket pipeline tunables (the /ws/audio-stream flow) ──
    # All previously-hardcoded constants from app/api/v1/routes/websocket.py,
    # now overridable from .env.
    # How often the idle-sweep loop seals quiet bubbles (turn_close), in seconds.
    idle_sweep_interval_s: float = Field(default=1.0, alias="IDLE_SWEEP_INTERVAL_S")
    # Drop a whole VAD segment when its peak amplitude is below this (mic noise floor).
    # Lowered 0.015 → 0.010 so genuinely soft / low-tone speech (a quiet patient,
    # a second voice) is no longer gated out as "too quiet". This is now safe
    # because hallucinations on faint audio are caught downstream by the Whisper
    # confidence gate (transcript_max_no_speech_prob / transcript_min_avg_logprob)
    # instead of by a blunt energy threshold. Raise it again if room/mic self-noise
    # starts producing phantom segments.
    segment_peak_noise_floor: float = Field(default=0.010, alias="SEGMENT_PEAK_NOISE_FLOOR")
    # ── Whisper confidence gate (anti-hallucination on faint speech) ──
    # A per-clip transcript is dropped only when BOTH hold: Whisper thinks the
    # clip is probably silence (no_speech_prob ≥ max) AND it decoded the words
    # with low confidence (avg_logprob ≤ min). Both-must-hold keeps real quiet
    # speech (which decodes confidently) while killing invented sentences.
    # Loosen (raise no_speech / lower avg_logprob) if quiet real speech is being
    # dropped; tighten if hallucinations still slip through.
    transcript_max_no_speech_prob: float = Field(default=0.65, alias="TRANSCRIPT_MAX_NO_SPEECH_PROB")
    transcript_min_avg_logprob: float = Field(default=-0.6, alias="TRANSCRIPT_MIN_AVG_LOGPROB")
    # Segments shorter than this many samples (at 16 kHz) are too short to diarize.
    min_segment_samples: int = Field(default=1600, alias="MIN_SEGMENT_SAMPLES")
    # Fuzzy similarity (0-1) at/above which two turns are treated as the same utterance.
    fuzzy_dup_threshold: float = Field(default=0.85, alias="FUZZY_DUP_THRESHOLD")
    # Merge consecutive same-speaker turns when their gap is under this (seconds)…
    merge_max_gap_s: float = Field(default=2.0, alias="MERGE_MAX_GAP_S")
    # …and both turns carry at least this role confidence (0-100).
    merge_min_conf: float = Field(default=50.0, alias="MERGE_MIN_CONF")
    # Per-stage timeouts (seconds).
    ws_receive_timeout_s: float = Field(default=120.0, alias="WS_RECEIVE_TIMEOUT_S")
    transcribe_timeout_s: float = Field(default=120.0, alias="TRANSCRIBE_TIMEOUT_S")
    rag_timeout_s: float = Field(default=30.0, alias="RAG_TIMEOUT_S")
    ollama_timeout_s: float = Field(default=300.0, alias="OLLAMA_TIMEOUT_S")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def allowed_origins(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
