import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SeparationResult:
    """Outcome of attempting to un-mix an overlapped audio segment."""
    separated: bool                      # True if we actually produced per-voice streams
    streams: List[np.ndarray] = field(default_factory=list)  # one waveform per voice
    overlap_ratio: float = 0.0           # fraction of the segment flagged as overlapped
    method: str = "none"                 # "sepformer" | "flagged" | "none"


class SeparationService:
    """Adaptive source separation for overlapping speech.

    Tier A: SpeechBrain Sepformer un-mixes the blended waveform into one stream
            per voice (loaded only if the package + model are available).
    Fallback: when Sepformer isn't available, we still DETECT overlap regions
            (via energy/spectral-flatness heuristics) and report the ratio so the
            caller can flag the turn as '[overlapping speech]' and attribute the
            dominant speaker — honest degradation instead of silent data loss.
    """

    def __init__(self, settings):
        self.settings = settings
        self.sample_rate = 16000
        self.backend = "flagged"     # "sepformer" once a model loads, else "flagged"
        self._model = None
        self._device = "cpu"
        if getattr(settings, "separation_enabled", True):
            self._init_backend()
        else:
            logger.info("Source separation disabled by settings")

    def _init_backend(self):
        want = getattr(self.settings, "separation_backend", "auto")
        if want not in ("auto", "sepformer"):
            logger.info("Separation backend set to flagging-only")
            return
        try:
            import torch
            from speechbrain.inference.separation import SepformerSeparation

            self._device = self._resolve_device()
            # speechbrain 1.x can't load directly on MPS — load on CPU then patch.
            load_device = "cpu" if self._device == "mps" else self._device
            # WSJ0-2mix model handles 2 simultaneous speakers; downloaded once.
            self._model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir="pretrained_models/sepformer-wsj02mix",
                run_opts={"device": load_device},
            )
            if self._device == "mps":
                from app.services._speechbrain_mps import force_speechbrain_to_mps
                self._model = force_speechbrain_to_mps(self._model)
            self.backend = "sepformer"
            logger.info(f"✅ Separation: Sepformer on {self._device}")
        except Exception as e:
            self.backend = "flagged"
            logger.warning(f"Sepformer unavailable ({e}); overlap will be flagged, not un-mixed")
            if want == "sepformer":
                logger.error("Sepformer explicitly requested but failed to load")

    def _resolve_device(self) -> str:
        """Resolve the requested device. MPS is now supported via the speechbrain workaround."""
        dev = getattr(self.settings, "diarization_device", "auto")
        if dev != "auto":
            return dev
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except Exception:
            return "cpu"

    # ── public API ──────────────────────────────────────────────
    def estimate_overlap_ratio(self, audio: np.ndarray, sr: int = 16000) -> float:
        """Heuristic estimate (0-1) of how much of the segment is overlapped speech.

        Overlapping voices raise spectral flatness (more noise-like, denser
        harmonics) and sustain higher energy. We score windows on both and
        return the fraction that look overlapped. Cheap, no model required.
        """
        try:
            import librosa
            if len(audio) < int(0.5 * sr):
                return 0.0
            frame, hop = 2048, 512
            flatness = librosa.feature.spectral_flatness(y=audio, n_fft=frame, hop_length=hop)[0]
            rms = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop)[0]
            if len(rms) == 0:
                return 0.0
            rms_norm = rms / (np.max(rms) + 1e-9)
            # A window looks overlapped when it's both energetic and spectrally flat.
            flat_thr = float(np.percentile(flatness, 60))
            overlapped = (flatness > max(flat_thr, 0.12)) & (rms_norm > 0.35)
            return float(np.mean(overlapped))
        except Exception as e:
            logger.warning(f"overlap estimation failed: {e}")
            return 0.0

    def maybe_separate(self, audio: np.ndarray, sr: int = 16000) -> SeparationResult:
        """Detect overlap; un-mix with Sepformer when available, else flag it."""
        ratio = self.estimate_overlap_ratio(audio, sr)
        min_ratio = getattr(self.settings, "overlap_min_ratio", 0.15)
        if ratio < min_ratio:
            return SeparationResult(separated=False, overlap_ratio=ratio, method="none")

        if self.backend == "sepformer" and self._model is not None:
            try:
                streams = self._separate_sepformer(audio, sr)
                if streams:
                    logger.info(f"🔀 Un-mixed overlap into {len(streams)} streams (ratio={ratio:.2f})")
                    return SeparationResult(True, streams, ratio, "sepformer")
            except Exception as e:
                logger.warning(f"Sepformer runtime error ({e}); flagging instead")

        # No un-mix available — report overlap so the caller can flag the turn.
        return SeparationResult(separated=False, overlap_ratio=ratio, method="flagged")

    def force_separate(self, audio: np.ndarray, sr: int = 16000) -> SeparationResult:
        """Un-mix a region ALREADY KNOWN to contain overlapping speakers.

        Unlike maybe_separate(), this skips the spectral-flatness ratio gate. Use
        it when an upstream signal (e.g. two diarized turns whose timestamps
        overlap) already establishes that the region is shared — the per-clip
        ratio heuristic would dilute a short (1-2s) overlap below threshold and
        miss it. Returns un-mixed streams when Sepformer is available; otherwise
        reports the region as 'flagged' so the caller can label it.
        """
        if self.backend == "sepformer" and self._model is not None:
            try:
                streams = self._separate_sepformer(audio, sr)
                if streams:
                    logger.info(f"🔀 Force-un-mixed known-overlap region into {len(streams)} streams")
                    return SeparationResult(True, streams, 1.0, "sepformer")
            except Exception as e:
                logger.warning(f"Sepformer force-separate error ({e}); flagging instead")
        return SeparationResult(separated=False, overlap_ratio=1.0, method="flagged")

    def _separate_sepformer(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        import torch
        import librosa

        # Sepformer-wsj02mix expects 8kHz input.
        target_sr = 8000
        a8 = librosa.resample(audio, orig_sr=sr, target_sr=target_sr) if sr != target_sr else audio
        mix = torch.from_numpy(a8).float().unsqueeze(0).to(self._device)
        est = self._model.separate_batch(mix)          # (batch, time, n_sources)
        est = est.squeeze(0).detach().cpu().numpy()      # (time, n_sources)

        # Resample each source back to 16k and measure its TRUE energy BEFORE any
        # normalization. Sepformer-wsj02mix always emits exactly 2 sources even
        # when the input has only ONE real voice — the extra source is a
        # low-energy phantom/residual. We must drop it here: if we peak-normalize
        # it to full scale (as below) and hand it to Whisper, Whisper hallucinates
        # fluent words nobody said and a phantom speaker identity is minted.
        raw = []
        for i in range(est.shape[-1]):
            src16 = librosa.resample(est[:, i], orig_sr=target_sr, target_sr=sr)
            rms = float(np.sqrt(np.mean(np.square(src16)))) if len(src16) else 0.0
            raw.append((src16, rms))

        loudest = max((r for _, r in raw), default=0.0)
        if loudest <= 0:
            return []

        # Keep only sources with real energy: above an absolute floor AND a
        # fraction of the loudest source. A genuine two-voice mix yields two
        # comparable-energy sources (both kept); a single-voice region yields one
        # strong + one weak source (only the strong kept → caller sees 1 stream
        # and treats the region as a single voice, no phantom).
        rel_floor = float(getattr(self.settings, "separation_stream_rel_floor", 0.30))
        abs_floor = 1e-3
        streams = []
        for src16, rms in raw:
            if rms >= abs_floor and rms >= rel_floor * loudest:
                peak = np.max(np.abs(src16)) + 1e-9
                streams.append((src16 / peak * 0.95).astype(np.float32))
            else:
                logger.info(
                    f"🚫 Dropped phantom Sepformer source (rms={rms:.4f} < "
                    f"{rel_floor:.2f}×{loudest:.4f}) — likely no real 2nd voice"
                )
        return streams
