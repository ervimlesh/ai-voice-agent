import logging
from typing import List, Optional, Tuple

import numpy as np

from app.models.speaker_profile import RegisteredSpeaker
from app.services.role_detector_service import SpeakerRole

logger = logging.getLogger(__name__)


class SpeakerRegistry:
    """Dynamic registry of speakers (S1..Sn) with sticky role assignment.

    Replaces the fixed doctor/patient slots so 3+ speakers (e.g. doctor +
    patient + relative) each get a stable identity across the whole
    conversation. Identity is matched on the embedding; roles are sticky to
    avoid per-turn flip-flop.

    One instance per WebSocket session. Reset on start_session / reset_history.
    """

    def __init__(self, settings):
        self.settings = settings
        self._speakers: List[RegisteredSpeaker] = []
        self._next = 1

    def reset(self):
        self._speakers.clear()
        self._next = 1
        logger.info("Speaker registry reset")

    # ── identity ────────────────────────────────────────────────
    def identify(
        self, embedding: Optional[List[float]], threshold: Optional[float] = None
    ) -> Tuple[str, bool]:
        """Match an embedding to a known speaker or register a new one.

        `threshold` overrides the configured cosine threshold for this call —
        used so neural (ECAPA) embeddings can apply their own tuned threshold
        without mutating shared settings.

        Returns (stable_speaker_id, is_new).
        """
        if embedding is None or len(embedding) == 0:
            # No usable voice signal → register as its own (unknown) identity.
            return self._register(np.zeros(1)), True

        emb = np.array(embedding, dtype=float)
        best_id, best_sim = None, -1.0
        for spk in self._speakers:
            sim = self._cos(emb, spk.embedding)
            if sim > best_sim:
                best_id, best_sim = spk.speaker_id, sim

        if threshold is None:
            threshold = getattr(self.settings, "speaker_match_threshold", 0.78)
        if best_id is not None and best_sim >= threshold:
            self._update(best_id, emb)
            return best_id, False
        return self._register(emb), True

    def _register(self, emb: np.ndarray) -> str:
        sid = f"S{self._next}"
        self._next += 1
        self._speakers.append(RegisteredSpeaker(speaker_id=sid, embedding=emb))
        logger.info(f"🆕 Registered speaker {sid} (total now {len(self._speakers)})")
        return sid

    def _update(self, sid: str, emb: np.ndarray):
        for spk in self._speakers:
            if spk.speaker_id == sid:
                spk.turn_count += 1
                if spk.embedding.shape == emb.shape:
                    a = 0.8  # running average keeps identity stable but adaptive
                    spk.embedding = a * spk.embedding + (1.0 - a) * emb
                else:
                    spk.embedding = emb
                return

    # ── role assignment (sticky) ────────────────────────────────
    def assign_role(
        self, sid: str, content_role: SpeakerRole, content_conf: float
    ) -> Tuple[str, float]:
        """Assign/maintain a speaker's role with anti-flip-flop logic.

        - Once a speaker's role is locked (high-confidence), keep it unless new
          evidence is both very strong AND disagrees.
        - 3rd+ speakers who aren't clearly Doctor/Patient become 'Relative'.
        """
        spk = self._get(sid)
        if spk is None:
            role = content_role.value if content_role != SpeakerRole.UNKNOWN else "Doctor"
            return role, content_conf or 50.0

        lock_thr = getattr(self.settings, "role_lock_confidence", 75.0)

        if spk.role_locked:
            if content_role.value not in (spk.role, "Unknown") and content_conf >= 90.0:
                spk.role, spk.role_confidence = content_role.value, content_conf
            return spk.role, spk.role_confidence

        if content_role != SpeakerRole.UNKNOWN and content_conf >= spk.role_confidence:
            spk.role = content_role.value
            spk.role_confidence = content_conf
            if content_conf >= lock_thr:
                spk.role_locked = True
        elif spk.role == "Unknown":
            # No content evidence yet. Assign a sensible default by registration
            # order so we NEVER surface "Unknown" to the UI: first speaker is the
            # Doctor, second the Patient, any further voice a Relative.
            idx = self._index(sid)
            spk.role = "Doctor" if idx == 0 else "Patient" if idx == 1 else "Relative"
            spk.role_confidence = max(spk.role_confidence, 50.0)

        return spk.role, spk.role_confidence

    def known_roles(self) -> dict:
        return {s.speaker_id: s.role for s in self._speakers}

    def _get(self, sid: str) -> Optional[RegisteredSpeaker]:
        return next((s for s in self._speakers if s.speaker_id == sid), None)

    def _index(self, sid: str) -> int:
        """0-based registration order of a speaker (0 = first registered)."""
        for i, s in enumerate(self._speakers):
            if s.speaker_id == sid:
                return i
        return len(self._speakers)

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return -1.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return -1.0
        return float(np.dot(a, b) / (na * nb))
