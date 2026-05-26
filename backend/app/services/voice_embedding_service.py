"""
Neural speaker-embedding service (voice fingerprinting).

Produces a high-dimensional, L2-normalized speaker embedding for a single audio
clip using SpeechBrain's ECAPA-TDNN model (trained on VoxCeleb). These
embeddings discriminate *who* is speaking far better than hand-crafted MFCC
statistics, which is what makes per-turn identity reliable even when several
people speak in the same segment (each separated stream gets its own
fingerprint).

Design:
- Loaded once as a singleton (see app.api.dependencies.get_voice_embedding_service).
- Auto-selects CUDA when available, else CPU. Works on both.
- Degrades gracefully: if the model can't load, `available` is False and the
  caller falls back to the lightweight MFCC embedding.
"""

import logging
import threading
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VoiceEmbeddingService:
    """Singleton wrapper around an ECAPA-TDNN speaker encoder."""

    def __init__(self, settings=None):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()
        self.device = self._resolve_device()
        self.embedding_dim: Optional[int] = None
        self._enabled = getattr(settings, "voice_embedding_enabled", True) if settings else True
        if self._enabled:
            self._load_model()
        else:
            logger.info("Voice embedding disabled by settings")

    # ── lifecycle ───────────────────────────────────────────────────────────
    def _resolve_device(self) -> str:
        want = getattr(self.settings, "voice_embedding_device", "auto") if self.settings else "auto"
        if want and want != "auto":
            return want
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_model(self):
        model_id = (
            getattr(self.settings, "voice_embedding_model", None)
            if self.settings else None
        ) or "speechbrain/spkrec-ecapa-voxceleb"
        try:
            # speechbrain>=1.0 moved the public API to `inference`; fall back to
            # the older `pretrained` namespace for compatibility.
            try:
                from speechbrain.inference import SpeakerRecognition
            except Exception:
                from speechbrain.pretrained import SpeakerRecognition

            logger.info(f"Loading ECAPA-TDNN voice embedding model on {self.device}...")
            self._model = SpeakerRecognition.from_hparams(
                source=model_id,
                savedir=f"pretrained_models/{model_id.split('/')[-1]}",
                run_opts={"device": self.device},
            )
            logger.info("✅ Voice embedding model ready (ECAPA-TDNN, neural fingerprints)")
        except Exception as e:
            logger.warning(
                f"Voice embedding model unavailable ({e}); "
                f"identity will fall back to lightweight MFCC embeddings"
            )
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def recommended_threshold(self) -> float:
        """Cosine-similarity match threshold tuned for ECAPA embeddings.

        Same-speaker clips typically score 0.45-0.95; different speakers 0.0-0.4.
        """
        return float(getattr(self.settings, "ecapa_match_threshold", 0.50)) if self.settings else 0.50

    # ── inference ─────────────────────────────────────────────────────────────
    def embed(self, audio: np.ndarray, sr: int = 16000) -> Optional[List[float]]:
        """Return an L2-normalized speaker embedding (list of floats), or None.

        None is returned on any failure or when the model is unavailable so the
        caller can decide how to degrade.
        """
        if self._model is None or audio is None or len(audio) == 0:
            return None

        try:
            import torch

            audio = np.asarray(audio, dtype=np.float32)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # Pad very short clips — ECAPA needs a little context to be stable.
            min_len = 8000  # 0.5 s at 16 kHz
            if len(audio) < min_len:
                audio = np.pad(audio, (0, min_len - len(audio)))

            wav = torch.from_numpy(audio).float().unsqueeze(0)
            if self.device == "cuda":
                wav = wav.cuda()

            with torch.no_grad():
                emb = self._model.encode_batch(wav)

            emb = emb.squeeze().detach().cpu().numpy()
            if emb.ndim > 1:
                emb = emb.flatten()

            norm = np.linalg.norm(emb)
            if norm == 0:
                return None
            emb = emb / norm
            self.embedding_dim = int(emb.shape[0])
            return emb.astype(np.float32).tolist()

        except Exception as e:
            logger.debug(f"Voice embedding extraction failed: {e}")
            return None
