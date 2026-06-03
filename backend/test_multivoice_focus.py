"""
Focused test for the doctor↔patient multi-voice changes:

  1. ROLE DETECTION DISABLED → roles come purely from speaker IDENTITY order
     (1st voice = Doctor, 2nd = Patient, 3rd+ = Relative) and stay STABLE across
     turns even when a later turn's words would have "looked like" the other role.
  2. OVERLAPPING / SIMULTANEOUS speakers each keep their OWN live bubble — an
     interruption by speaker B does not hijack speaker A's open bubble.
  3. SLOW-SPEECH VAD settings are relaxed so a deliberate speaker isn't cut off.

Run:  .venv/bin/python test_multivoice_focus.py
"""
import sys
from types import SimpleNamespace

from app.services.role_detector_service import SpeakerRole
from app.services.speaker_registry_service import SpeakerRegistry
from app.services.conversation_state import ConversationState
from app.services.voice_activity_detector import VoiceActivityDetector


def _ok(msg): print(f"  ✅ {msg}")
def _fail(msg): print(f"  ❌ {msg}"); sys.exit(1)


def test_speaker_order_roles_with_detection_disabled():
    print("1) Role detection OFF → stable speaker-order roles")
    # Mirror the websocket sentinel: role detection returns UNKNOWN/0 for every
    # clip, so assign_role falls back to registration order.
    settings = SimpleNamespace(role_lock_confidence=80.0, speaker_match_threshold=0.78)
    reg = SpeakerRegistry(settings)

    # Mirror the real pipeline: identify() registers the voice FIRST, then
    # assign_role() runs with the role-detection sentinel (UNKNOWN, 0.0).
    # Three mutually-distinct embeddings → three distinct speakers.
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.0, 1.0, 0.0]
    emb_c = [0.0, 0.0, 1.0]

    def speak(emb):
        sid, _is_new = reg.identify(emb)
        return reg.assign_role(sid, SpeakerRole.UNKNOWN, 0.0)

    # First distinct voice speaks several times -> always Doctor.
    for _ in range(3):
        role, _ = speak(emb_a)
    if role != "Doctor":
        _fail(f"first speaker should be Doctor, got {role}")
    _ok("1st voice → Doctor, stable across 3 turns")

    # Second distinct voice -> Patient (no content heuristic involved anymore).
    role2, _ = speak(emb_b)
    if role2 != "Patient":
        _fail(f"second speaker should be Patient, got {role2}")
    _ok("2nd voice → Patient")

    # Third voice -> Relative.
    role3, _ = speak(emb_c)
    if role3 != "Relative":
        _fail(f"third speaker should be Relative, got {role3}")
    _ok("3rd voice → Relative")

    # Roles never flip on re-speak (no content can override an order-assigned role).
    again, _ = speak(emb_a)
    if again != "Doctor":
        _fail(f"first voice role flipped to {again}")
    _ok("Roles never flip-flop on re-speak")


def test_overlapping_speakers_keep_own_bubbles():
    print("2) Overlapping speakers each keep their own bubble")
    state = ConversationState()
    t = 1000.0

    # Doctor starts a sentence.
    state.coalesce_or_open(
        {"speaker_id": "S1", "role": "Doctor", "role_confidence": 50.0,
         "text": "How are you feeling", "start": 0.0, "end": 1.5, "language": "en"},
        now=t)
    # Patient interrupts mid-sentence (overlap) — must NOT hijack S1's bubble.
    state.coalesce_or_open(
        {"speaker_id": "S2", "role": "Patient", "role_confidence": 50.0,
         "text": "My chest hurts", "start": 1.2, "end": 2.4, "overlap": True,
         "language": "en"},
        now=t + 0.1)
    # Doctor resumes — should CONTINUE the same S1 bubble, not open a new one.
    evs = state.coalesce_or_open(
        {"speaker_id": "S1", "role": "Doctor", "role_confidence": 50.0,
         "text": "today", "start": 2.5, "end": 3.0, "language": "en"},
        now=t + 0.2)

    if len(state.open_turns) != 2:
        _fail(f"expected 2 simultaneous open bubbles, got {len(state.open_turns)}")
    _ok("Both speakers have a live bubble at once (overlap preserved)")

    resume_type = evs[-1]["type"]
    if resume_type != "turn_update":
        _fail(f"doctor resume should extend bubble (turn_update), got {resume_type}")
    if "today" not in state.open_turns["S1"].text:
        _fail("doctor's resumed text was not appended to the same bubble")
    _ok("Doctor's resume continued the SAME bubble across the interruption")
    if not state.open_turns["S2"].overlap:
        _fail("patient turn lost its overlap flag")
    _ok("Patient turn carries the ⚠ overlap flag for the UI")


def test_slow_speech_vad_settings():
    print("3) Slow-speech VAD settings relaxed")
    vad = VoiceActivityDetector()
    if vad.min_silence_duration_ms < 600:
        _fail(f"silence window {vad.min_silence_duration_ms}ms too short for slow speech")
    _ok(f"min_silence_duration_ms = {vad.min_silence_duration_ms}ms "
        f"(was 250ms — no longer cuts slow speakers off mid-sentence)")
    if vad.speech_threshold > 0.30:
        _fail(f"speech_threshold {vad.speech_threshold} too high for quiet/slow speech")
    _ok(f"speech_threshold = {vad.speech_threshold} (triggers on quieter/slower speech)")


if __name__ == "__main__":
    test_speaker_order_roles_with_detection_disabled()
    test_overlapping_speakers_keep_own_bubbles()
    test_slow_speech_vad_settings()
    print("\n🎉 ALL CHECKS PASSED — multi-voice focus changes verified.")
