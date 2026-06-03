"""
Behavior tests for ConversationState — the bubble lifecycle that decides
whether a new turn extends an existing bubble or opens a fresh one.

These tests pin down the contract for the two requirements that have been
churning in production:

  R1. Rapid sequential alternation — when speaker B starts before A finishes,
      A's growing bubble should be sealed at A's last word and a brand-new
      bubble should open for B. The two must NEVER merge into one bubble,
      regardless of how close in time they arrive on the server.

  R2a. Continuous same-speaker speech — when one speaker keeps talking with
       small (<5s audio) pauses, all of it stays in ONE growing bubble.

  R2b. Stop-and-restart — when the same speaker stops, falls silent for
       longer than the audio-coalesce window, and then speaks again, the
       resumption opens a NEW bubble, not an extension of the old one.

  R3.  Sweep-idle — bubbles with no activity past the wall-time coalesce
       window get closed by the periodic sweep so the UI seals them even
       without a follow-up utterance.

The tests deliberately do NOT touch real audio, Whisper, or pyannote — they
exercise the deterministic state machine directly. Audio-pipeline regressions
that affect this state machine WILL fail these tests; acoustic regressions
(e.g. pyannote can't tell two voices apart) are out of scope here and need
real-audio integration tests instead.
"""
from __future__ import annotations

import time

import pytest

from app.services.conversation_state import (
    AUDIO_COALESCE_GAP_S,
    COALESCE_GAP_S,
    ConversationState,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def make_turn(
    *,
    speaker_id: str,
    role: str,
    text: str,
    start: float,
    end: float,
    confidence: float = 90.0,
    overlap: bool = False,
    language: str = "en",
    is_new_speaker: bool = False,
) -> dict:
    """Build a turn dict in the exact shape websocket.py emits to coalesce_or_open."""
    return {
        "speaker_id": speaker_id,
        "role": role,
        "role_confidence": confidence,
        "text": text,
        "start": start,
        "end": end,
        "language": language,
        "overlap": overlap,
        "is_new_speaker": is_new_speaker,
    }


def drive(state: ConversationState, turns: list[dict], wall_times: list[float] | None = None) -> list[dict]:
    """Feed turns to coalesce_or_open, return the flat list of emitted events."""
    if wall_times is None:
        wall_times = [time.time() + i * 0.001 for i in range(len(turns))]
    assert len(wall_times) == len(turns), "wall_times length must match turns"
    events: list[dict] = []
    for t, w in zip(turns, wall_times):
        events.extend(state.coalesce_or_open(t, now=w))
    return events


def event_types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def texts_per_bubble(events: list[dict]) -> dict[str, str]:
    """Final text per turn_id after replaying all events."""
    out: dict[str, str] = {}
    for e in events:
        if e["type"] in ("turn_new", "turn_update"):
            out[e["turn_id"]] = e["text"]
        # turn_close also carries final_text but we already saw it via update
    return out


# ──────────────────────────────────────────────────────────────────────
# R1. Rapid sequential alternation (different speakers)
# ──────────────────────────────────────────────────────────────────────

class TestRapidSequentialAlternation:
    """When B starts before A ends, A and B must end up in SEPARATE bubbles
    even though both turns reach the server with ~0 wall-time gap."""

    def test_a_then_b_back_to_back_opens_two_bubbles(self):
        state = ConversationState()
        turns = [
            make_turn(speaker_id="S1", role="Doctor",
                      text="Did you take any medicine?",
                      start=0.0, end=2.0),
            make_turn(speaker_id="S2", role="Patient",
                      text="No I didn't",
                      start=1.8, end=3.5),  # B starts before A's end → overlap in audio time
        ]
        # Same wall time for both — they arrived in the same Python call.
        wall = time.time()
        events = drive(state, turns, wall_times=[wall, wall])

        # Expect: two distinct turn_new events, NO turn_update merging across speakers.
        new_events = [e for e in events if e["type"] == "turn_new"]
        assert len(new_events) == 2, f"Expected 2 new bubbles, got {len(new_events)}: {events}"
        speakers = {e["speaker_id"] for e in new_events}
        assert speakers == {"S1", "S2"}, f"Wrong speaker IDs: {speakers}"

        # Each bubble keeps its own text — no cross-contamination.
        bubbles = texts_per_bubble(events)
        for tid, text in bubbles.items():
            assert "Did you take" in text or "No I didn't" in text
            # Critically: no bubble contains BOTH phrases.
            assert not ("Did you take" in text and "No I didn't" in text), \
                f"Bubble cross-contamination: {text!r}"

    def test_three_way_rapid_overlap_a_b_a_keeps_a_in_one_bubble(self):
        """A interrupted by B, then A resumes within the audio-gap window.
        A's resume should EXTEND A's bubble (it's still A's turn from the
        listener's POV); B gets its own bubble in the middle."""
        state = ConversationState()
        wall = time.time()
        turns = [
            make_turn(speaker_id="S1", role="Doctor",
                      text="So as I was saying,", start=0.0, end=1.5),
            make_turn(speaker_id="S2", role="Patient",
                      text="sorry to interrupt,", start=1.2, end=2.6),
            make_turn(speaker_id="S1", role="Doctor",
                      text="take rest and drink fluids.", start=2.4, end=4.5),
        ]
        events = drive(state, turns, wall_times=[wall, wall + 0.01, wall + 0.02])

        new_events = [e for e in events if e["type"] == "turn_new"]
        update_events = [e for e in events if e["type"] == "turn_update"]

        # S1 should have one bubble (opened first, then extended on resume);
        # S2 should have one bubble of its own.
        s1_new = [e for e in new_events if e["speaker_id"] == "S1"]
        s2_new = [e for e in new_events if e["speaker_id"] == "S2"]
        s1_updates = [e for e in update_events if e["speaker_id"] == "S1"]

        assert len(s1_new) == 1, f"S1 should open exactly one bubble, got {len(s1_new)}"
        assert len(s2_new) == 1, f"S2 should open exactly one bubble, got {len(s2_new)}"
        assert len(s1_updates) >= 1, "S1's resume must EXTEND the existing bubble, not open a new one"

        # The final S1 bubble must contain BOTH of S1's utterances joined.
        s1_final = (s1_updates[-1] if s1_updates else s1_new[-1])["text"]
        assert "So as I was saying" in s1_final
        assert "take rest" in s1_final

        # S2's bubble must NOT contain S1's words.
        s2_text = s2_new[0]["text"]
        assert "So as I was saying" not in s2_text
        assert "take rest" not in s2_text


# ──────────────────────────────────────────────────────────────────────
# R2a. Continuous same-speaker speech across small gaps
# ──────────────────────────────────────────────────────────────────────

class TestContinuousSpeaker:

    def test_consecutive_same_speaker_within_audio_gap_extends_one_bubble(self):
        """Two same-speaker turns 2s apart in audio time stay in ONE bubble."""
        state = ConversationState()
        turns = [
            make_turn(speaker_id="S1", role="Patient",
                      text="I have a fever.", start=0.0, end=2.0),
            make_turn(speaker_id="S1", role="Patient",
                      text="And a headache.", start=4.0, end=6.0),  # 2s audio gap
        ]
        wall = time.time()
        events = drive(state, turns, wall_times=[wall, wall + 0.5])

        new_events = [e for e in events if e["type"] == "turn_new"]
        update_events = [e for e in events if e["type"] == "turn_update"]

        assert len(new_events) == 1, f"Continuous speech should be ONE bubble, got {len(new_events)} new"
        assert len(update_events) >= 1, "Second utterance should EXTEND the first"

        final_text = update_events[-1]["text"]
        assert "I have a fever" in final_text
        assert "And a headache" in final_text


# ──────────────────────────────────────────────────────────────────────
# R2b. Stop-and-restart (same speaker, large audio gap)
# ──────────────────────────────────────────────────────────────────────

class TestStopAndRestart:

    def test_same_speaker_audio_gap_above_threshold_opens_new_bubble(self):
        """Two same-speaker utterances separated by >AUDIO_COALESCE_GAP_S audio
        time must open two separate bubbles."""
        state = ConversationState()
        big_gap = AUDIO_COALESCE_GAP_S + 1.0  # safely past the audio coalesce window
        turns = [
            make_turn(speaker_id="S1", role="Patient",
                      text="First utterance.", start=0.0, end=2.0),
            make_turn(speaker_id="S1", role="Patient",
                      text="Second utterance after a long pause.",
                      start=2.0 + big_gap, end=2.0 + big_gap + 2.0),
        ]
        # Process them in the same Python call (wall time ≈ 0). This is the
        # exact case that the old wall-time-only coalesce got wrong.
        wall = time.time()
        events = drive(state, turns, wall_times=[wall, wall + 0.001])

        new_events = [e for e in events if e["type"] == "turn_new"]
        close_events = [e for e in events if e["type"] == "turn_close"]

        assert len(new_events) == 2, \
            f"Audio gap > {AUDIO_COALESCE_GAP_S}s must open a new bubble, got {len(new_events)}"
        assert len(close_events) >= 1, "Old bubble must be sealed before the new one opens"

        # Each bubble carries only its own utterance.
        texts = [e["text"] for e in new_events]
        assert any("First utterance" in t and "Second" not in t for t in texts)
        assert any("Second utterance" in t and "First" not in t for t in texts)

    def test_same_speaker_long_segment_with_audio_gaps_splits_correctly(self):
        """Reproduces the production bug from the user's log: one VAD segment,
        13 diarized turns, same speaker_id S2 appearing at audio times
        5.4–17.8, 25.1–27.5, 41.8–47.97, 70.7–79.0. These must NOT all
        collapse into one bubble — they are four separate utterances."""
        state = ConversationState()
        turns = [
            make_turn(speaker_id="S2", role="Patient",
                      text="A", start=5.40, end=17.83),
            make_turn(speaker_id="S2", role="Patient",
                      text="B", start=25.07, end=27.47),  # gap 7.24s
            make_turn(speaker_id="S2", role="Patient",
                      text="C", start=41.78, end=47.97),  # gap 14.31s
            make_turn(speaker_id="S2", role="Patient",
                      text="D", start=70.72, end=78.99),  # gap 22.75s
        ]
        wall = time.time()
        events = drive(state, turns, wall_times=[wall + i * 0.001 for i in range(len(turns))])
        new_events = [e for e in events if e["type"] == "turn_new"]
        assert len(new_events) == 4, \
            f"Each large-gap S2 utterance must be its own bubble; got {len(new_events)}"


# ──────────────────────────────────────────────────────────────────────
# R3. Sweep-idle closes bubbles after wall-time silence
# ──────────────────────────────────────────────────────────────────────

class TestSweepIdle:

    def test_sweep_idle_closes_bubble_past_wall_gap(self):
        state = ConversationState()
        wall = time.time()
        state.coalesce_or_open(
            make_turn(speaker_id="S1", role="Patient",
                      text="Hello", start=0.0, end=1.0),
            now=wall,
        )
        assert len(state.open_turns) == 1

        # Sweep BEFORE the wall gap → nothing happens.
        sweep_msgs = state.sweep_idle(now=wall + 1.0)
        assert sweep_msgs == [], "Sweep should not close bubble within wall gap"
        assert len(state.open_turns) == 1

        # Sweep AFTER the wall gap → bubble closes.
        sweep_msgs = state.sweep_idle(now=wall + COALESCE_GAP_S + 0.5)
        assert len(sweep_msgs) == 1
        assert sweep_msgs[0]["type"] == "turn_close"
        assert sweep_msgs[0]["speaker_id"] == "S1"
        assert len(state.open_turns) == 0

    def test_close_all_on_reset(self):
        state = ConversationState()
        state.coalesce_or_open(
            make_turn(speaker_id="S1", role="Doctor", text="x", start=0, end=1)
        )
        state.coalesce_or_open(
            make_turn(speaker_id="S2", role="Patient", text="y", start=0, end=1)
        )
        msgs = state.close_all()
        types = [m["type"] for m in msgs]
        assert types == ["turn_close", "turn_close"]
        assert state.open_turns == {}


# ──────────────────────────────────────────────────────────────────────
# R2 (simultaneous). Two voices talking at the same time → un-mixed by
# Sepformer into separate streams → each stream is a separate turn with
# the same audio time range but different speaker_id. ConversationState
# must place them in TWO independent bubbles, never merge them.
# ──────────────────────────────────────────────────────────────────────

class TestSimultaneousOverlap:

    def test_separated_overlap_streams_get_independent_bubbles(self):
        state = ConversationState()
        # Both turns occupy 0-3s of audio (true talk-over), but
        # arrive with different speaker_ids because Sepformer un-mixed them.
        turns = [
            make_turn(speaker_id="S1", role="Doctor",
                      text="What brings you in today?",
                      start=0.0, end=3.0, overlap=True),
            make_turn(speaker_id="S2", role="Patient",
                      text="My chest is hurting since morning.",
                      start=0.0, end=3.0, overlap=True),
        ]
        wall = time.time()
        events = drive(state, turns, wall_times=[wall, wall + 0.001])

        new_events = [e for e in events if e["type"] == "turn_new"]
        assert len(new_events) == 2, \
            f"Two simultaneous speakers must produce two bubbles, got {len(new_events)}"

        # Both bubbles are flagged as overlap (UI hint that the moment was talk-over).
        assert all(e.get("overlap") for e in new_events), \
            "Both bubbles from separated streams should carry the overlap flag"

        # Each carries only its own speaker's text.
        s1 = next(e for e in new_events if e["speaker_id"] == "S1")
        s2 = next(e for e in new_events if e["speaker_id"] == "S2")
        assert "chest" not in s1["text"], "Doctor's bubble must not contain patient's words"
        assert "What brings" not in s2["text"], "Patient's bubble must not contain doctor's words"


# ──────────────────────────────────────────────────────────────────────
# Role flip safety — same voice newly classified should close-and-open
# ──────────────────────────────────────────────────────────────────────

class TestRoleFlip:

    def test_same_speaker_role_flip_seals_old_and_opens_new(self):
        """If the same speaker_id arrives with a different role (rare but
        happens during role recalibration), the old bubble must close so
        the UI doesn't visually re-label a sealed bubble."""
        state = ConversationState()
        wall = time.time()
        ev1 = state.coalesce_or_open(
            make_turn(speaker_id="S1", role="Patient", text="hello", start=0, end=1),
            now=wall,
        )
        ev2 = state.coalesce_or_open(
            make_turn(speaker_id="S1", role="Doctor", text="take rest", start=1, end=2),
            now=wall + 0.01,
        )
        types = event_types(ev1) + event_types(ev2)
        assert types.count("turn_new") == 2
        assert types.count("turn_close") == 1
