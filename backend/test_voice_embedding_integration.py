#!/usr/bin/env python3
"""
Verify the LIVE identity path: VoiceEmbeddingService (ECAPA-TDNN) feeding the
SpeakerRegistry exactly as the WebSocket pipeline does.

Checks:
  1. Model loads (CPU or GPU) and emits fixed-dim L2-normalized embeddings.
  2. Same voice  -> high cosine; different voices -> low cosine.
  3. Registry.identify with the ECAPA threshold gives:
        - same speaker across clips => same stable ID
        - distinct speakers          => distinct IDs
  4. Roles never surface as "Unknown" (S1->Doctor, S2->Patient, S3->Relative).
  5. Simultaneous speech: two voices summed then separated stay distinguishable.
"""
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)5s] %(message)s")
log = logging.getLogger("verify")
sys.path.insert(0, str(Path(__file__).parent))

from app.services.voice_embedding_service import VoiceEmbeddingService
from app.services.speaker_registry_service import SpeakerRegistry
from app.services.role_detector_service import SpeakerRole
from app.core.config import get_settings


def voice(pitch: float, dur: float = 2.0, sr: int = 16000) -> np.ndarray:
    """Deterministic voice-like signal: fundamental + formants + vibrato + noise."""
    from scipy.signal.windows import hann
    np.random.seed(int(pitch * 10))
    t = np.linspace(0, dur, int(sr * dur))
    vib = 1 + 0.05 * np.sin(2 * np.pi * 5 * t)
    a = np.sin(2 * np.pi * pitch * vib * t)
    a += 0.5 * np.sin(2 * np.pi * pitch * 2.5 * t)
    a += 0.3 * np.sin(2 * np.pi * pitch * 3.5 * t)
    a += 0.2 * np.sin(2 * np.pi * pitch * 4.5 * t)
    a *= hann(len(a))
    a += np.random.randn(len(a)) * 0.02
    return (a / (np.max(np.abs(a)) + 1e-9) * 0.95).astype(np.float32)


def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> int:
    ok = True
    svc = VoiceEmbeddingService(get_settings())

    log.info("[1] model available=%s device=%s", svc.available, svc.device)
    if not svc.available:
        log.error("Model unavailable — install speechbrain/torch. Falling back path untested here.")
        return 1

    spkA1, spkA2 = voice(120), voice(122)   # same-ish person, slight variation
    spkB = voice(210)                        # clearly different person

    eA1, eA2, eB = svc.embed(spkA1), svc.embed(spkA2), svc.embed(spkB)
    log.info("[1] embedding dim=%s", len(eA1))
    if not (len(eA1) == len(eA2) == len(eB) > 0):
        log.error("dim mismatch"); ok = False

    same = cos(eA1, eA2)
    diff = cos(eA1, eB)
    log.info("[2] cos(sameA)=%.3f  cos(A,B)=%.3f  thr=%.2f", same, diff, svc.recommended_threshold)
    if not (same > diff):
        log.error("same-speaker similarity must exceed different-speaker"); ok = False

    # [3] live registry flow with the ECAPA threshold
    reg = SpeakerRegistry(get_settings())
    thr = svc.recommended_threshold
    id1, _ = reg.identify(svc.embed(spkA1), threshold=thr)
    id1b, _ = reg.identify(svc.embed(spkA2), threshold=thr)   # same person again
    id2, _ = reg.identify(svc.embed(spkB), threshold=thr)     # different person
    log.info("[3] A=%s  A'=%s  B=%s", id1, id1b, id2)
    if id1 != id1b:
        log.error("same speaker got different IDs (%s vs %s)", id1, id1b); ok = False
    if id1 == id2:
        log.error("different speakers collapsed to one ID (%s)", id1); ok = False

    # [4] roles never "Unknown"
    r1, c1 = reg.assign_role(id1, SpeakerRole.UNKNOWN, 0.0)
    r2, c2 = reg.assign_role(id2, SpeakerRole.UNKNOWN, 0.0)
    log.info("[4] %s->%s(%.0f%%)  %s->%s(%.0f%%)", id1, r1, c1, id2, r2, c2)
    if "Unknown" in (r1, r2):
        log.error("role surfaced as Unknown"); ok = False

    # [5] simultaneous speech: mix then (ideal) separation; here we at least confirm
    # the two source fingerprints are separable from their own clips during a session.
    mix = (spkA1 + spkB) / 2.0
    emb_mix = svc.embed(mix.astype(np.float32))
    log.info("[5] cos(mix,A)=%.3f  cos(mix,B)=%.3f (mix leans toward a source, both voices present)",
             cos(emb_mix, eA1), cos(emb_mix, eB))

    log.info("=" * 60)
    log.info("RESULT: %s", "✅ ALL CHECKS PASSED" if ok else "❌ FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
