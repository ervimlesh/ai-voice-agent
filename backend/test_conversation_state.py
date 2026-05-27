"""
Unit tests for ConversationState — the continuous-speaker bubble coalescer.

Headline scenarios under test:
  - Same speaker keeps talking → ONE growing bubble across multiple segments
  - Speaker B interrupts while A's bubble is open → B gets a new bubble, A's
    bubble survives the interruption and continues on A's next utterance
  - Idle past COALESCE_GAP_S → bubble closes; same speaker's next utterance
    opens a fresh bubble
  - Role flip on the same voice → close + open a new bubble
  - Per-speaker LLM history threads stay separate

Run:
    pytest test_conversation_state.py -v
"""
import os
# HF_TOKEN is read from the environment (.env / shell). Never hardcode it.
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

import pytest

from app.schemas.chat import ChatMessage
from app.services.conversation_state import (
    ConversationState,
    COALESCE_GAP_S,
)


def make_turn(
    *,
    speaker_id: str,
    role: str = "Patient",
    role_confidence: float = 90.0,
    text: str = "hello",
    start: float = 0.0,
    end: float = 1.0,
    overlap: bool = False,
    is_new_speaker: bool = False,
    language: str = "en",
) -> dict:
    return {
        "speaker_id": speaker_id,
        "role": role,
        "role_confidence": role_confidence,
        "text": text,
        "start": start,
        "end": end,
        "overlap": overlap,
        "is_new_speaker": is_new_speaker,
        "language": language,
    }


# ───────────────────────────────────────────────────────────────────────
# 1. Single-speaker continuation
# ───────────────────────────────────────────────────────────────────────

class TestSingleSpeakerContinuation:
    def test_first_utterance_is_turn_new(self):
        state = ConversationState()
        out = state.coalesce_or_open(make_turn(speaker_id="S1", text="hello"), now=100.0)
        assert len(out) == 1
        assert out[0]["type"] == "turn_new"
        assert out[0]["text"] == "hello"
        assert "turn_id" in out[0]

    def test_continuation_same_speaker_within_gap_emits_turn_update(self):
        state = ConversationState()
        first = state.coalesce_or_open(make_turn(speaker_id="S1", text="hello"), now=100.0)
        second = state.coalesce_or_open(
            make_turn(speaker_id="S1", text="how are you", start=2.0, end=3.0),
            now=100.0 + COALESCE_GAP_S - 1.0,  # well inside the coalesce window
        )
        assert second[0]["type"] == "turn_update"
        assert second[0]["turn_id"] == first[0]["turn_id"]
        # Text was glued together
        assert "hello" in second[0]["text"]
        assert "how are you" in second[0]["text"]

    def test_gap_exceeded_closes_and_opens_new_bubble(self):
        state = ConversationState()
        first = state.coalesce_or_open(make_turn(speaker_id="S1", text="hello"), now=100.0)
        # COALESCE_GAP_S + 2s later — gap exceeded regardless of the constant value
        events = state.coalesce_or_open(
            make_turn(speaker_id="S1", text="back again"),
            now=100.0 + COALESCE_GAP_S + 2.0,
        )
        types = [e["type"] for e in events]
        assert types == ["turn_close", "turn_new"]
        assert events[0]["turn_id"] == first[0]["turn_id"]
        assert events[1]["turn_id"] != first[0]["turn_id"]


# ───────────────────────────────────────────────────────────────────────
# 2. The headline scenario — A talks, B interrupts, A continues
# ───────────────────────────────────────────────────────────────────────

class TestOverlapPreservesPrimarySpeaker:
    def test_interruption_does_not_break_primary_speakers_bubble(self):
        state = ConversationState()
        a1 = state.coalesce_or_open(
            make_turn(speaker_id="A", role="Doctor", text="how are you feeling"),
            now=100.0,
        )
        # Speaker B interrupts (overlap=True per Sepformer flag)
        b = state.coalesce_or_open(
            make_turn(speaker_id="B", role="Patient", text="my chest hurts", overlap=True),
            now=100.5,
        )
        # Speaker A resumes 1s later — well within COALESCE_GAP_S
        a2 = state.coalesce_or_open(
            make_turn(speaker_id="A", role="Doctor", text="today"),
            now=101.5,
        )

        # B opens its own NEW bubble
        assert b[0]["type"] == "turn_new"
        assert b[0]["turn_id"] != a1[0]["turn_id"]

        # A's bubble survived and continued (turn_update on the original turn_id)
        assert a2[0]["type"] == "turn_update"
        assert a2[0]["turn_id"] == a1[0]["turn_id"]
        assert "today" in a2[0]["text"]

    def test_two_simultaneous_open_turns_during_overlap(self):
        state = ConversationState()
        state.coalesce_or_open(make_turn(speaker_id="A", role="Doctor"), now=100.0)
        state.coalesce_or_open(
            make_turn(speaker_id="B", role="Patient", overlap=True), now=100.2
        )
        assert "A" in state.open_turns
        assert "B" in state.open_turns
        assert state.open_turns["A"].turn_id != state.open_turns["B"].turn_id


# ───────────────────────────────────────────────────────────────────────
# 3. Role flip on the same voice
# ───────────────────────────────────────────────────────────────────────

class TestRoleFlip:
    def test_role_flip_closes_and_opens_new(self):
        state = ConversationState()
        first = state.coalesce_or_open(
            make_turn(speaker_id="S1", role="Patient", text="hello"), now=100.0
        )
        flip = state.coalesce_or_open(
            make_turn(speaker_id="S1", role="Doctor", text="hi there"), now=100.5
        )
        types = [e["type"] for e in flip]
        assert types == ["turn_close", "turn_new"]
        assert flip[0]["turn_id"] == first[0]["turn_id"]
        assert flip[1]["turn_id"] != first[0]["turn_id"]
        assert flip[1]["role"] == "Doctor"


# ───────────────────────────────────────────────────────────────────────
# 4. Idle sweep
# ───────────────────────────────────────────────────────────────────────

class TestIdleSweep:
    def test_sweep_closes_idle_bubbles(self):
        state = ConversationState()
        first = state.coalesce_or_open(make_turn(speaker_id="A"), now=100.0)
        # Sweep past the coalesce window so the idle bubble is sealed
        closed = state.sweep_idle(now=100.0 + COALESCE_GAP_S + 2.0)
        assert len(closed) == 1
        assert closed[0]["type"] == "turn_close"
        assert closed[0]["turn_id"] == first[0]["turn_id"]
        assert "A" not in state.open_turns

    def test_sweep_keeps_active_bubbles(self):
        state = ConversationState()
        state.coalesce_or_open(make_turn(speaker_id="A"), now=100.0)
        closed = state.sweep_idle(now=101.0)  # only 1s idle
        assert closed == []
        assert "A" in state.open_turns

    def test_close_all_emits_close_for_each(self):
        state = ConversationState()
        state.coalesce_or_open(make_turn(speaker_id="A"), now=100.0)
        state.coalesce_or_open(make_turn(speaker_id="B"), now=100.1)
        events = state.close_all(now=200.0)
        assert len(events) == 2
        assert all(e["type"] == "turn_close" for e in events)
        assert state.open_turns == {}


# ───────────────────────────────────────────────────────────────────────
# 5. Per-speaker LLM history
# ───────────────────────────────────────────────────────────────────────

class TestPerSpeakerHistory:
    def test_histories_are_isolated(self):
        state = ConversationState()
        state.append_history("A", ChatMessage(role="user", content="A says hi"))
        state.append_history("B", ChatMessage(role="user", content="B says hi"))
        assert len(state.get_history("A")) == 1
        assert len(state.get_history("B")) == 1
        assert state.get_history("A")[0].content == "A says hi"
        assert state.get_history("B")[0].content == "B says hi"

    def test_history_is_trimmed_at_limit(self):
        from app.services.conversation_state import PER_SPEAKER_HISTORY_LIMIT
        state = ConversationState()
        for i in range(PER_SPEAKER_HISTORY_LIMIT + 5):
            state.append_history("A", ChatMessage(role="user", content=f"msg {i}"))
        assert len(state.get_history("A")) == PER_SPEAKER_HISTORY_LIMIT

    def test_legacy_history_is_combined(self):
        state = ConversationState()
        state.append_history("A", ChatMessage(role="user", content="A1"))
        state.append_history("B", ChatMessage(role="user", content="B1"))
        # Legacy history sees both, in order
        contents = [m.content for m in state.legacy_history]
        assert contents == ["A1", "B1"]


# ───────────────────────────────────────────────────────────────────────
# 5b. Unified multi-voice history
# ───────────────────────────────────────────────────────────────────────

class TestUnifiedHistory:
    """Multi-voice mode: when the user is talking to ChatGPT voice (or any
    other voice agent), the LLM should receive BOTH speakers' utterances
    as context, not just the user's own."""

    def test_unified_history_returns_legacy_history_reference(self):
        state = ConversationState()
        state.append_unified_user("[Doctor]: Hello, hi.")
        state.append_unified_assistant("Greeting acknowledged.")
        state.append_unified_user("[Patient]: I have a fever.")
        hist = state.get_unified_history()
        contents = [(m.role, m.content) for m in hist]
        assert contents == [
            ("user", "[Doctor]: Hello, hi."),
            ("assistant", "Greeting acknowledged."),
            ("user", "[Patient]: I have a fever."),
        ]

    def test_unified_history_skips_empty(self):
        state = ConversationState()
        state.append_unified_user("")
        state.append_unified_user("   ")
        state.append_unified_assistant("")
        assert state.get_unified_history() == []

    def test_unified_history_respects_limit(self):
        from app.services.conversation_state import UNIFIED_HISTORY_LIMIT
        state = ConversationState()
        for i in range(UNIFIED_HISTORY_LIMIT + 5):
            state.append_unified_user(f"msg {i}")
        assert len(state.get_unified_history()) == UNIFIED_HISTORY_LIMIT


# ───────────────────────────────────────────────────────────────────────
# 5c. Multi-voice segment text composition
# ───────────────────────────────────────────────────────────────────────

class TestBuildSegmentUserText:
    """The LLM-facing user message for one VAD segment.

    Doctor + Patient in the same segment must produce a role-tagged
    transcript so the assistant can answer with full multi-voice context.
    A single-speaker segment must read naturally (no tagging clutter).
    """

    def test_single_speaker_no_tagging(self):
        out = ConversationState.build_segment_user_text(
            [{"role": "Patient", "text": "I have a fever"}]
        )
        assert out == "I have a fever"

    def test_single_role_two_turns_joined(self):
        out = ConversationState.build_segment_user_text([
            {"role": "Patient", "text": "I have a fever"},
            {"role": "Patient", "text": "since last night"},
        ])
        # Same role → no tagging, concatenated naturally
        assert out == "I have a fever since last night"
        assert "[Patient]" not in out

    def test_multi_role_gets_tagged(self):
        out = ConversationState.build_segment_user_text([
            {"role": "Doctor", "text": "Hello, hi."},
            {"role": "Patient", "text": "I'm suffering from fever"},
        ])
        # Two distinct roles → LLM needs to see who said what
        assert "[Doctor]: Hello, hi." in out
        assert "[Patient]: I'm suffering from fever" in out

    def test_overlap_tag_is_stripped(self):
        out = ConversationState.build_segment_user_text([
            {"role": "Doctor", "text": "[overlapping speech] hello"},
            {"role": "Patient", "text": "fever"},
        ])
        # The sentinel that flags overlap on the wire MUST NOT pollute the LLM input
        assert "[overlapping speech]" not in out
        assert "[Doctor]: hello" in out

    def test_empty_turns_filtered(self):
        out = ConversationState.build_segment_user_text([
            {"role": "Doctor", "text": ""},
            {"role": "Patient", "text": "  "},
        ])
        assert out == ""

    def test_user_scenario_chatgpt_voice_in_context(self):
        """The exact bug the user reported: when ChatGPT voice (Doctor) and
        the user's voice (Patient) are both in one segment, the LLM input
        must include BOTH — not just the user's words."""
        chatgpt_text = "Based on your fever, you should rest"
        user_text = "But I still feel weak"
        out = ConversationState.build_segment_user_text([
            {"role": "Doctor", "text": chatgpt_text},
            {"role": "Patient", "text": user_text},
        ])
        assert chatgpt_text in out, "ChatGPT's voice MUST be in LLM context"
        assert user_text in out, "User's voice MUST be in LLM context"
        assert out.index(f"[Doctor]: {chatgpt_text}") < out.index(f"[Patient]: {user_text}")


# ───────────────────────────────────────────────────────────────────────
# 6. Reset
# ───────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_everything(self):
        state = ConversationState()
        state.coalesce_or_open(make_turn(speaker_id="A"), now=100.0)
        state.append_history("A", ChatMessage(role="user", content="hi"))
        state.reset()
        assert state.open_turns == {}
        assert state.per_speaker_history == {}
        assert state.legacy_history == []


# ───────────────────────────────────────────────────────────────────────
# 7. End-to-end scenario — the user's stated requirement
# ───────────────────────────────────────────────────────────────────────

class TestUserScenario:
    """The literal scenario from the user's brief:

       "When one user speaks then at the same time second user also speaks,
        show that already one user is on same context as he was already
        speaking. And on the other hand the second user that overlapped
        shows in next context. So we don't need multiple contexts for
        continuous same user who is speaking continuously."
    """

    def test_continuous_doctor_with_patient_interruption(self):
        state = ConversationState()
        events: list[dict] = []

        # Doctor starts talking
        events += state.coalesce_or_open(
            make_turn(speaker_id="DOC", role="Doctor", text="Tell me about your symptoms"),
            now=100.0,
        )
        # Doctor continues
        events += state.coalesce_or_open(
            make_turn(speaker_id="DOC", role="Doctor", text="when did it start"),
            now=101.0,
        )
        # Patient cuts in
        events += state.coalesce_or_open(
            make_turn(speaker_id="PAT", role="Patient", text="three days ago", overlap=True),
            now=101.5,
        )
        # Doctor resumes shortly after
        events += state.coalesce_or_open(
            make_turn(speaker_id="DOC", role="Doctor", text="and is the pain sharp or dull"),
            now=103.0,
        )

        # Two distinct turn_ids: one for Doctor (continued), one for Patient
        new_events = [e for e in events if e["type"] == "turn_new"]
        update_events = [e for e in events if e["type"] == "turn_update"]
        close_events = [e for e in events if e["type"] == "turn_close"]

        assert len(new_events) == 2, "Exactly two bubbles should have opened: Doctor and Patient"
        assert len(close_events) == 0, "No bubble should have closed mid-flow"
        assert len(update_events) >= 2, "Doctor's bubble should have been updated multiple times"

        doctor_new = next(e for e in new_events if e["speaker_id"] == "DOC")
        # Every Doctor update must reference the SAME turn_id
        for u in update_events:
            if u["speaker_id"] == "DOC":
                assert u["turn_id"] == doctor_new["turn_id"], (
                    "Doctor's bubble lost continuity when Patient interrupted"
                )

        # Doctor's final text contains all three Doctor utterances
        final_doc_text = update_events[-1]["text"] if update_events[-1]["speaker_id"] == "DOC" else doctor_new["text"]
        for u in reversed(update_events):
            if u["speaker_id"] == "DOC":
                final_doc_text = u["text"]
                break
        assert "Tell me" in final_doc_text
        assert "when did" in final_doc_text
        assert "sharp or dull" in final_doc_text
