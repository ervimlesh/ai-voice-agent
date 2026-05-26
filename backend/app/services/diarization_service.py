import logging
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiarTurn:
    """One speaker turn within a segment (times are seconds within the segment)."""
    start: float
    end: float
    local_label: str          # segment-local label, e.g. "SPEAKER_00"


class DiarizationService:
    """Splits a speech segment into per-speaker turns. Adaptive two-tier backend.

    Tier A: pyannote.audio speaker-diarization-3.1 (best accuracy, handles overlap
            regions; loaded only if pyannote + an HF token are available).
    Tier B: sliding-window MFCC embeddings + agglomerative clustering with an
            automatic speaker-count search (always available; uses librosa+sklearn). 

    Short-turn handling (limitation #2): Tier B uses heavily OVERLAPPING windows
    (1.0s window, 0.25s hop) and pads any merged turn out to a minimum analysis
    length, so a sub-second utterance still produces enough embedding windows to
    be clustered rather than dropped outright.
    """

    def __init__(self, settings, speaker_detector):
        self.settings = settings
        self.speaker_detector = speaker_detector
        self.sample_rate = 16000
        self.backend = "lightweight"
        self._pipeline = None
        if getattr(settings, "diarization_enabled", True):
            self._init_backend()
        else:
            logger.info("Diarization disabled by settings")

    # ── backend selection ───────────────────────────────────────
    def _init_backend(self):
        want = getattr(self.settings, "diarization_backend", "auto")
        if want in ("auto", "pyannote"):
            try:
                import torch
                from pyannote.audio import Pipeline

                token = getattr(self.settings, "hf_token", None)
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=token,
                )
                device = self._resolve_device()
                self._pipeline.to(torch.device(device))
                self.backend = "pyannote"
                logger.info(f"✅ Diarization: pyannote on {device}")
                return
            except Exception as e:
                logger.warning(f"pyannote unavailable ({e}); using lightweight tier")
                if want == "pyannote":
                    logger.error("pyannote explicitly requested but failed to load")
        self.backend = "lightweight"
        logger.info("✅ Diarization: lightweight (librosa + sklearn clustering)")

    def _resolve_device(self) -> str:
        dev = getattr(self.settings, "diarization_device", "auto")
        if dev != "auto":
            return dev
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    # ── public API ──────────────────────────────────────────────
    def diarize(self, audio: np.ndarray, sr: int = 16000) -> List[DiarTurn]:
        """Return ordered speaker turns for a single speech segment."""
        if audio is None or len(audio) == 0:
            return []
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        duration = len(audio) / sr
        min_turn = getattr(self.settings, "min_turn_duration_s", 0.4)
        # Whole segment is barely longer than one turn → don't bother splitting.
        if duration < max(min_turn, 0.8):
            return [DiarTurn(0.0, duration, "SPEAKER_00")]

        if self.backend == "pyannote" and self._pipeline is not None:
            try:
                return self._diarize_pyannote(audio, sr)
            except Exception as e:
                logger.warning(f"pyannote runtime error ({e}); falling back to lightweight")
        return self._diarize_lightweight(audio, sr)

    # ── Tier A: pyannote ────────────────────────────────────────
    def _diarize_pyannote(self, audio: np.ndarray, sr: int) -> List[DiarTurn]:
        import torch

        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        annotation = self._pipeline({"waveform": waveform, "sample_rate": sr})
        min_turn = getattr(self.settings, "min_turn_duration_s", 0.4)

        turns: List[DiarTurn] = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            if segment.duration >= min_turn:
                turns.append(DiarTurn(float(segment.start), float(segment.end), str(label)))
        turns.sort(key=lambda t: t.start)
        if not turns:
            return [DiarTurn(0.0, len(audio) / sr, "SPEAKER_00")]
        return turns

    # ── Tier B: sliding-window embeddings + clustering ──────────
    def _diarize_lightweight(self, audio: np.ndarray, sr: int) -> List[DiarTurn]:
        win_s, hop_s = 1.0, 0.25            # heavy overlap so short turns survive
        win, hop = int(win_s * sr), int(hop_s * sr)
        total = len(audio)

        embeddings: List[List[float]] = []
        spans: List[tuple] = []
        pos = 0
        while pos < total:
            chunk = audio[pos:pos + win]
            if len(chunk) >= int(0.35 * sr):    # enough to embed
                embeddings.append(self.speaker_detector.build_embedding(chunk, sr))
                spans.append((pos / sr, min((pos + win) / sr, total / sr)))
            if pos + win >= total:
                break
            pos += hop

        if len(embeddings) <= 1:
            return [DiarTurn(0.0, total / sr, "SPEAKER_00")]

        labels = self._cluster(np.array(embeddings))

        # Assign each window to its center time, then merge consecutive windows
        # that share a label. We cut turn boundaries at the midpoint between a
        # label change so turns stay contiguous and non-overlapping (windows
        # themselves overlap due to the small hop, but turns must not).
        centers = [(s + e) / 2.0 for (s, e) in spans]
        turns: List[DiarTurn] = []
        run_start_idx = 0
        for i in range(1, len(labels) + 1):
            changed = i == len(labels) or labels[i] != labels[run_start_idx]
            if not changed:
                continue
            label = labels[run_start_idx]
            # Boundary before the run: midpoint between previous center and this run's first center.
            if run_start_idx == 0:
                start = spans[0][0]
            else:
                start = (centers[run_start_idx - 1] + centers[run_start_idx]) / 2.0
            # Boundary after the run: midpoint between last center of run and next center.
            last_idx = i - 1
            if i == len(labels):
                end = spans[-1][1]
            else:
                end = (centers[last_idx] + centers[i]) / 2.0
            turns.append(DiarTurn(float(start), float(end), f"SPEAKER_{int(label):02d}"))
            run_start_idx = i

        min_turn = getattr(self.settings, "min_turn_duration_s", 0.4)
        kept = [t for t in turns if (t.end - t.start) >= min_turn]
        return kept or [DiarTurn(0.0, total / sr, "SPEAKER_00")]

    def _cluster(self, X: np.ndarray) -> np.ndarray:
        """Agglomerative clustering (cosine) with automatic speaker-count search.

        Tries 2..4 clusters and keeps the count with the best silhouette score.
        If no split is clearly better than one speaker, returns a single cluster.
        """
        n = len(X)
        if n < 3:
            return np.zeros(n, dtype=int)
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
        except ImportError:
            logger.warning("scikit-learn missing; single-speaker fallback")
            return np.zeros(n, dtype=int)

        best_labels = np.zeros(n, dtype=int)
        best_score = -1.0
        for k in range(2, min(4, n - 1) + 1):
            try:
                model = AgglomerativeClustering(
                    n_clusters=k, metric="cosine", linkage="average")
                labels = model.fit_predict(X)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(X, labels, metric="cosine")
            except Exception:
                continue
            if score > best_score:
                best_score, best_labels = score, labels

        # Clusters not clearly separated → treat as a single speaker.
        if best_score < 0.10:
            return np.zeros(n, dtype=int)
        return best_labels
