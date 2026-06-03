"""
Per-WebSocket conversation state — coalesces continuous same-speaker turns into
ONE growing bubble and gives each speaker their own LLM history thread.

Design rationale (the bug the single-`open_turn` approach hit):
  Doctor:  "How are you feeling—"   ┐
  Patient:        "My chest hurts"  │  overlap
  Doctor:  "—today?"                ┘
With one global `open_turn`, the Patient's interruption would overwrite the
Doctor's open bubble; Doctor's resume becomes a fresh bubble (wrong — it should
continue the prior one). We track open turns *per speaker* so both stay alive
simultaneously during overlap.

A bubble closes when its speaker has been silent past `COALESCE_GAP_S`, or
their detected role flips on the same speaker_id.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.schemas.chat import ChatMessage

# Wall-clock gap: bubble idle past this in real time → seal it (idle sweep).
COALESCE_GAP_S = 8.0
# Audio-time gap: if a new turn's audio start is this far past the existing
# bubble's audio end, treat it as a new utterance — even if the server
# processed them in the same Python call (wall-time gap ≈ 0).
#
# This prevents a long VAD segment (e.g. 80s of rapid doctor-patient
# back-and-forth) from collapsing all of one speaker's separated turns into
# one giant bubble. Audio gap of 5s is the "obvious new utterance" boundary —
# below that we still coalesce (continuous speech with thought pauses).
AUDIO_COALESCE_GAP_S = 5.0
PER_SPEAKER_HISTORY_LIMIT = 20
UNIFIED_HISTORY_LIMIT = 30


@dataclass
class OpenTurn:
    """A bubble that is still accepting more text from its speaker."""
    turn_id: str
    speaker_id: str
    role: str
    role_confidence: float
    text: str
    segment_start: float
    segment_end: float
    language: str
    last_activity_wall: float
    overlap: bool = False


@dataclass
class ConversationState:
    """All bubble + history state for one WebSocket connection."""
    open_turns: Dict[str, OpenTurn] = field(default_factory=dict)
    per_speaker_history: Dict[str, List[ChatMessage]] = field(default_factory=dict)
    legacy_history: List[ChatMessage] = field(default_factory=list)
    coalesce_gap_s: float = COALESCE_GAP_S

    # ──────────────────────────────────────────────────────────────────
    # Bubble coalescing
    # ──────────────────────────────────────────────────────────────────

    def coalesce_or_open(self, turn: dict, now: Optional[float] = None) -> List[dict]:
        """Decide whether `turn` extends an existing bubble or opens a new one.

        Returns a list of websocket messages to send (in order). Each message is
        a dict ready to ship — caller does `await ws.send_json(msg)` for each.

        The continuation rule (per-speaker, NOT global):
          - the speaker has an open bubble, AND
          - role hasn't flipped on that same voice, AND
          - the speaker's own last activity was within COALESCE_GAP_S
        """
        if now is None:
            now = time.time()
        speaker_id = turn["speaker_id"]
        out: List[dict] = []

        existing = self.open_turns.get(speaker_id)
        same_role = existing is not None and existing.role == turn["role"]
        within_wall_gap = (
            existing is not None
            and (now - existing.last_activity_wall) < self.coalesce_gap_s
        )
        # Audio-time gap: turns inside one long segment all process at near-
        # zero wall-time gap, so wall_gap alone collapses them. Use the audio
        # timestamps to detect that two same-speaker pieces are actually far
        # apart in the recording and should be separate bubbles.
        audio_gap = (
            (turn.get("start", 0.0) - existing.segment_end)
            if existing is not None else float("inf")
        )
        within_audio_gap = existing is not None and audio_gap < AUDIO_COALESCE_GAP_S
        within_gap = within_wall_gap and within_audio_gap

        if existing is not None and same_role and within_gap:
            new_text = (turn["text"] or "").strip()
            if new_text:
                existing.text = _join_text(existing.text, new_text)
            existing.segment_end = turn["end"]
            existing.role_confidence = max(existing.role_confidence, turn["role_confidence"])
            existing.last_activity_wall = now
            if turn.get("overlap"):
                existing.overlap = True
            out.append({
                "type": "turn_update",
                "turn_id": existing.turn_id,
                "speaker_id": existing.speaker_id,
                "role": existing.role,
                "role_confidence": existing.role_confidence,
                "text": existing.text,
                "end": existing.segment_end,
                "overlap": existing.overlap,
                "language": existing.language,
                "timestamp": now,
            })
            return out

        # Role flipped on the same voice — close old, open new.
        if existing is not None and not same_role:
            out.append(self._close_message(existing, now))
            del self.open_turns[speaker_id]
        # Otherwise: same_role but wall_gap or audio_gap exceeded — close and
        # open a fresh bubble for this new utterance.
        elif existing is not None and not within_gap:
            out.append(self._close_message(existing, now))
            del self.open_turns[speaker_id]

        new_turn = OpenTurn(
            turn_id=str(uuid.uuid4()),
            speaker_id=speaker_id,
            role=turn["role"],
            role_confidence=turn["role_confidence"],
            text=(turn["text"] or "").strip(),
            segment_start=turn["start"],
            segment_end=turn["end"],
            language=turn.get("language", "en"),
            last_activity_wall=now,
            overlap=bool(turn.get("overlap")),
        )
        self.open_turns[speaker_id] = new_turn
        out.append({
            "type": "turn_new",
            "turn_id": new_turn.turn_id,
            "speaker_id": new_turn.speaker_id,
            "role": new_turn.role,
            "role_confidence": new_turn.role_confidence,
            "text": new_turn.text,
            "start": new_turn.segment_start,
            "end": new_turn.segment_end,
            "is_new_speaker": turn.get("is_new_speaker", False),
            "overlap": new_turn.overlap,
            "language": new_turn.language,
            "timestamp": now,
        })
        return out

    def sweep_idle(self, now: Optional[float] = None) -> List[dict]:
        """Close any bubbles whose speaker has been silent past COALESCE_GAP_S.
        Called periodically by the websocket so the frontend can seal bubbles
        without having to wait for the next utterance from somebody else."""
        if now is None:
            now = time.time()
        out: List[dict] = []
        for sid, ot in list(self.open_turns.items()):
            if now - ot.last_activity_wall >= self.coalesce_gap_s:
                out.append(self._close_message(ot, now))
                del self.open_turns[sid]
        return out

    def close_all(self, now: Optional[float] = None) -> List[dict]:
        """Used on session reset / disconnect."""
        if now is None:
            now = time.time()
        out = [self._close_message(ot, now) for ot in self.open_turns.values()]
        self.open_turns.clear()
        return out

    def _close_message(self, ot: OpenTurn, now: float) -> dict:
        return {
            "type": "turn_close",
            "turn_id": ot.turn_id,
            "speaker_id": ot.speaker_id,
            "final_text": ot.text,
            "end": ot.segment_end,
            "timestamp": now,
        }

    # ──────────────────────────────────────────────────────────────────
    # Per-speaker LLM history
    # ──────────────────────────────────────────────────────────────────

    def get_history(self, speaker_id: str) -> List[ChatMessage]:
        """Return the LLM message history thread for this speaker (mutable)."""
        return self.per_speaker_history.setdefault(speaker_id, [])

    def append_history(self, speaker_id: str, msg: ChatMessage) -> None:
        hist = self.get_history(speaker_id)
        hist.append(msg)
        if len(hist) > PER_SPEAKER_HISTORY_LIMIT:
            del hist[: len(hist) - PER_SPEAKER_HISTORY_LIMIT]
        # Mirror into legacy_history so any consumer that still expects a single
        # rolling history (e.g. old clients) keeps working.
        self.legacy_history.append(msg)
        if len(self.legacy_history) > PER_SPEAKER_HISTORY_LIMIT:
            del self.legacy_history[: len(self.legacy_history) - PER_SPEAKER_HISTORY_LIMIT]

    # ──────────────────────────────────────────────────────────────────
    # Unified multi-voice history (sees EVERY speaker, including the
    # "other" voice when the user is talking to another voice agent like
    # ChatGPT voice). Per-speaker history alone would hide the other
    # voice from the LLM, so the assistant would reply without context.
    # ──────────────────────────────────────────────────────────────────

    def get_unified_history(self) -> List[ChatMessage]:
        """Return the chronological multi-speaker conversation history."""
        return self.legacy_history

    @staticmethod
    def build_segment_user_text(turns: List[dict]) -> str:
        """Compose the LLM user-message text for one VAD segment.

        Multi-voice rule:
          - if the segment contains only one speaker (or one role), return the
            plain concatenated text — no tagging, reads naturally.
          - if the segment contains multiple roles (e.g. ChatGPT voice +
            user voice both in one segment), return a role-tagged transcript
            like "[Doctor]: …\n[Patient]: …" so the LLM knows who said what.

        Each turn dict is expected to have keys `role` and `text`. The
        sentinel "[overlapping speech] " prefix that the pipeline adds for
        flagged overlaps is stripped before composing.
        """
        OVERLAP_TAG = "[overlapping speech] "
        clean: List[dict] = []
        for t in turns:
            text = (t.get("text") or "").replace(OVERLAP_TAG, "").strip()
            role = t.get("role") or "Unknown"
            if text:
                clean.append({"role": role, "text": text})
        if not clean:
            return ""
        unique_roles = {t["role"] for t in clean}
        if len(unique_roles) == 1:
            return " ".join(t["text"] for t in clean)
        return "\n".join(f"[{t['role']}]: {t['text']}" for t in clean)

    def append_unified_user(self, text: str) -> None:
        """Append a tagged user message (possibly multi-speaker) to the
        shared history. Caller is responsible for tagging the text with
        speaker roles when more than one speaker is present in the segment."""
        if not text.strip():
            return
        self.legacy_history.append(ChatMessage(role="user", content=text))
        if len(self.legacy_history) > UNIFIED_HISTORY_LIMIT:
            del self.legacy_history[: len(self.legacy_history) - UNIFIED_HISTORY_LIMIT]

    def append_unified_assistant(self, text: str) -> None:
        if not text.strip():
            return
        self.legacy_history.append(ChatMessage(role="assistant", content=text))
        if len(self.legacy_history) > UNIFIED_HISTORY_LIMIT:
            del self.legacy_history[: len(self.legacy_history) - UNIFIED_HISTORY_LIMIT]

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.open_turns.clear()
        self.per_speaker_history.clear()
        self.legacy_history.clear()


def _join_text(prev: str, addition: str) -> str:
    """Glue two transcript chunks together with sensible spacing/punctuation."""
    prev_clean = prev.rstrip()
    add_clean = addition.lstrip()
    if not prev_clean:
        return add_clean
    if not add_clean:
        return prev_clean
    # If the previous chunk already ends with terminal punctuation, just space-join.
    if prev_clean.endswith((".", "!", "?", "…", "。", "؟", "।")):
        return f"{prev_clean} {add_clean}"
    # Otherwise add a period to separate the two thought units.
    return f"{prev_clean}. {add_clean}"
