import asyncio
import io
import json
import logging
import re
import tempfile
import time
import struct
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.dependencies import (
    get_whisper_service,
    get_ollama_service,
    get_rag_service,
    get_speaker_detector,
    get_diarization_service,
    get_separation_service,
    get_voice_embedding_service,
)
from app.services.voice_activity_detector import VoiceActivityDetector, VADEvent
from app.services.role_detector_service import RoleDetectorService, SpeakerRole
from app.services.speaker_registry_service import SpeakerRegistry
from app.services.conversation_state import ConversationState
from app.core.config import get_settings
from app.schemas.chat import ChatMessage

# How often to sweep for idle bubbles and emit turn_close events.
IDLE_SWEEP_INTERVAL_S = 1.0

# ── Audio-energy thresholds (single source of truth) ──────────────────
# Multi-voice setup notes:
#   • Headset mic capture is quiet: own voice rms ≈ 0.008–0.014,
#     peak ≈ 0.06–0.15.
#   • A second voice (ChatGPT through laptop speakers / phone speaker held
#     near the mic) arrives even quieter, typically peak ≈ 0.02–0.06.
#   • Mic noise floor with no signal is peak < ~0.012.
#
# So we gate ONLY at the segment level on `peak` (not rms), and only against
# the noise floor. Per-clip rms gating is intentionally disabled — it would
# silently drop the other voice. Hallucination filtering is content-based
# (see HALLUCINATION_SUBSTRINGS below and whisper_service.is_whisper_hallucination).
SEGMENT_PEAK_NOISE_FLOOR = 0.015     # drop whole segment if peak < this
MIN_SEGMENT_SAMPLES = 1600           # < 0.1s at 16 kHz: too short to diarize

# Hallucination phrases Whisper emits on silence/noise. Anything containing
# one of these as a substring is dropped, regardless of energy or speaker.
HALLUCINATION_SUBSTRINGS = (
    "thanks for watching",
    "thank you for watching",
    "thanks for listening",
    "thank you for listening",
    "please subscribe",
    "like and subscribe",
    "see you next time",
    "see you in the next",
    "thank you so much for watching",
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared VAD instance per connection is created in the handler.
# Whisper, Ollama, diarization and separation are singletons from dependencies.
# The SpeakerRegistry is per-connection (resets between conversations).


def _is_garbage_transcript(text: str) -> bool:
    """
    Detect ONLY extremely obvious garbage/noise (pure repetition with no other content).

    Much more lenient: we let Ollama decide how to handle borderline cases.
    Only reject if transcript is COMPLETELY repetitive with no semantic content.

    Handles:
    - Pure empty strings
    - Single characters repeated 10+ times (like "aaaaaaaaaa")
    - 95%+ identical tokens (extreme repetition)
    """
    stripped = text.strip()

    # Pattern A: Empty after stripping - only real empty
    if not stripped:
        return True

    # Pattern D: Pure punctuation or single characters - only if truly empty of alphanumeric
    alpha_only = re.sub(r'[\s\W]+', '', stripped, flags=re.UNICODE)
    if len(alpha_only) == 0:  # Completely empty of alphanumeric
        return True

    # Character-level extreme repetition (catches "aaaaaaaaaa" - 10+ of same char)
    chars = re.sub(r'\s+', '', stripped)
    if len(set(chars)) == 1 and len(chars) > 10:
        logger.info("🗑️ Garbage filter: detected extreme character repetition (single char 10+ times)")
        return True

    # Low character diversity over a long string is hallucinated noise, e.g.
    # "र्ध्र्ध्र्ध्…", "ༀༀༀༀ…", "☂☂☂…". No real utterance in any supported
    # language packs 15+ characters into ≤4 distinct symbols.
    if len(chars) >= 15 and len(set(chars)) <= 4:
        logger.info("🗑️ Garbage filter: low character diversity over long string")
        return True

    # Repeated short cycle: the text is essentially one short unit repeated
    # (e.g. "र्ध्" ×12). Try small periods and see if they reconstruct the string.
    if len(chars) >= 12:
        for period in range(1, 5):
            unit = chars[:period]
            reps = len(chars) // period + 1
            reconstructed = (unit * reps)[:len(chars)]
            matches = sum(1 for a, b in zip(chars, reconstructed) if a == b)
            if matches / len(chars) > 0.8:
                logger.info(f"🗑️ Garbage filter: repeated {period}-char cycle '{unit}'")
                return True

    # High symbol/dingbat ratio. Whisper hallucinations on noise often emit
    # symbol-category glyphs (☂ ✿ ◌ emoji …); normal speech has almost none.
    symbolish = sum(
        1 for c in chars
        if unicodedata.category(c) in ("So", "Sk", "Sm", "Cf", "Co", "Cn")
    )
    if chars and symbolish / len(chars) > 0.3:
        logger.info("🗑️ Garbage filter: high symbol/dingbat ratio")
        return True

    # Pattern B: Normalize comma separators
    normalized = re.sub(r'[、。，,،]+', ' ', stripped)
    tokens = [unicodedata.normalize('NFC', t) for t in normalized.split() if t.strip()]

    if not tokens:
        return True

    # Reject heavy token repetition ("yes yes yes yes …"). Tightened to 0.8 so we
    # catch hallucinated loops without dropping natural emphasis.
    if len(tokens) >= 5:
        counter = Counter(tokens)
        most_common_token, most_common_count = counter.most_common(1)[0]
        repetition_ratio = most_common_count / len(tokens)

        if repetition_ratio > 0.8:
            logger.info(f"🗑️ Garbage filter: detected token repetition ({repetition_ratio*100:.0f}% of '{most_common_token}')")
            return True

    return False


def decode_audio_bytes(raw_bytes: bytes, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """
    Decode raw PCM int16 LE bytes into float32 numpy array.
    Frontend sends raw PCM 16-bit signed little-endian at 16 kHz mono.
    """
    try:
        if len(raw_bytes) < 2:
            return None
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32
    except Exception as e:
        logger.error(f"Error decoding audio bytes: {e}")
        return None


def audio_segment_to_wav_bytes(audio_float32: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert float32 numpy audio to WAV bytes for Whisper"""
    audio_int16 = (audio_float32 * 32767).astype(np.int16)
    buf = io.BytesIO()
    # Write WAV header
    num_samples = len(audio_int16)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))  # chunk size
    buf.write(struct.pack('<H', 1))   # PCM format
    buf.write(struct.pack('<H', 1))   # mono
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', sample_rate * 2))  # byte rate
    buf.write(struct.pack('<H', 2))   # block align
    buf.write(struct.pack('<H', 16))  # bits per sample
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(audio_int16.tobytes())
    return buf.getvalue()


async def _safe_send_json(websocket: WebSocket, data: dict) -> bool:
    """Safely send JSON, returning False if connection is closed."""
    try:
        await websocket.send_json(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


async def _generate_fallback_suggestions(ollama_service, speaker_input: str, speaker_role: str = "Patient") -> list[str]:
    """
    Fallback suggestion generator when RAG database has no matching results.
    Uses Ollama to intelligently generate relevant follow-up questions a DOCTOR
    could ask next — regardless of whether the doctor or the patient is speaking.
    """
    try:
        if speaker_role == "Doctor":
            context_line = f'Doctor just said: "{speaker_input}"'
            guidance = (
                "Generate ONLY 3 concise, medical follow-up questions the doctor could ask next "
                "to deepen the clinical picture (history, differential, red flags). One per line, no numbering."
            )
        else:
            context_line = f'Patient just said: "{speaker_input}"'
            guidance = (
                "Generate ONLY 3 concise, medical follow-up questions the doctor should ask the patient. "
                "One per line, no numbering. Make them relevant to what the patient mentioned and useful for diagnosis."
            )

        suggestion_prompt = f"""{context_line}

{guidance}

Example format:
How long have you had this?
Is the pain constant or intermittent?
Have you tried any treatments?

Now generate 3 questions."""

        messages = [
            {
                "role": "system",
                "content": "You are a medical assistant helping doctors. Generate helpful follow-up questions."
            },
            {
                "role": "user",
                "content": suggestion_prompt
            }
        ]

        response = await ollama_service._chat(messages)

        # Parse response into list of questions
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        suggestions = questions[:3]  # Take first 3

        if suggestions:
            print(f"💡 Fallback Suggestions (from Ollama): {suggestions}")

        return suggestions

    except Exception as e:
        logger.warning(f"Error generating fallback suggestions: {e}")
        print(f"⚠️ Could not generate fallback suggestions: {e}")
        return []


async def _process_speech_segment_with_signal(
    websocket: WebSocket,
    audio_segment_bytes: bytes,
    whisper_service,
    ollama_service,
    state: ConversationState,
    vad: VoiceActivityDetector,
    processing_lock: asyncio.Lock,
    speaker_detector,
    role_detector,
    registry: "SpeakerRegistry",
    diarizer,
    separator,
    rag_service,
    voice_embedder,
):
    """Wrapper that processes segment and signals when done."""
    async with processing_lock:
        await _process_speech_segment(
            websocket,
            audio_segment_bytes,
            whisper_service,
            ollama_service,
            state,
            vad,
            speaker_detector,
            role_detector,
            registry,
            diarizer,
            separator,
            rag_service,
            voice_embedder,
        )
        # Signal frontend to continue listening after processing
        await _safe_send_json(websocket, {
            "type": "ready_for_next",
            "timestamp": time.time()
        })


@router.websocket("/ws/audio-stream")
async def audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for continuous audio streaming with Silero VAD.

    Protocol:
    - Client sends binary frames: raw PCM int16 LE audio at 16kHz mono
    - Client sends text frames: JSON commands (e.g., {"type": "start_session"})
    - Server sends text frames: JSON events (VAD events, transcripts, AI responses)
    """
    await websocket.accept()

    vad = VoiceActivityDetector()
    whisper_service = get_whisper_service()
    ollama_service = get_ollama_service()
    rag_service = get_rag_service()
    speaker_detector = get_speaker_detector()
    role_detector = RoleDetectorService()
    diarizer = get_diarization_service()
    separator = get_separation_service()
    voice_embedder = get_voice_embedding_service()

    # Per-connection conversation state: open bubbles + per-speaker LLM history.
    state = ConversationState()
    session_active = True
    processing_lock = asyncio.Lock()
    background_tasks = set()

    # Per-conversation dynamic speaker registry (handles 3+ speakers).
    registry = SpeakerRegistry(get_settings())

    async def _idle_sweep_loop():
        """Emit turn_close for any bubble idle past COALESCE_GAP_S."""
        try:
            while True:
                await asyncio.sleep(IDLE_SWEEP_INTERVAL_S)
                for msg in state.sweep_idle():
                    if not await _safe_send_json(websocket, msg):
                        return
        except asyncio.CancelledError:
            return

    sweep_task = asyncio.create_task(_idle_sweep_loop())

    print("WebSocket connected: audio-stream")

    try:
        logger.info("WebSocket connection established")
        print("🟢 WebSocket connection established")
        await websocket.send_json({
            "type": "connected",
            "message": "Audio stream connected. Send PCM16 audio at 16kHz mono.",
            "timestamp": time.time()
        })

        while session_active:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=120.0)
            except asyncio.TimeoutError:
                # Send keepalive
                if not await _safe_send_json(websocket, {"type": "keepalive", "timestamp": time.time()}):
                    break
                continue

            if "text" in message:
                # Handle JSON commands
                try:
                    cmd = json.loads(message["text"])
                    cmd_type = cmd.get("type", "")

                    if cmd_type == "start_session":
                        vad.reset_states()
                        for msg in state.close_all():
                            await _safe_send_json(websocket, msg)
                        state.reset()
                        registry.reset()  # fresh speaker identities for new conversation
                        await websocket.send_json({
                            "type": "session_started",
                            "timestamp": time.time()
                        })
                        print("Session started via WebSocket")

                    elif cmd_type == "end_session":
                        session_active = False
                        await websocket.send_json({
                            "type": "session_ended",
                            "timestamp": time.time()
                        })
                        print("Session ended via WebSocket")

                    elif cmd_type == "reset_history":
                        for msg in state.close_all():
                            await _safe_send_json(websocket, msg)
                        state.reset()
                        vad.reset_states()
                        registry.reset()  # Reset speaker identities for new conversation
                        await websocket.send_json({
                            "type": "history_reset",
                            "timestamp": time.time()
                        })

                    elif cmd_type == "set_history":
                        # Allow client to sync history (legacy single-thread variant).
                        raw_history = cmd.get("history", [])
                        state.legacy_history = [ChatMessage(**m) for m in raw_history]
                        await websocket.send_json({
                            "type": "history_synced",
                            "history_length": len(state.legacy_history),
                            "timestamp": time.time()
                        })

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON command",
                        "timestamp": time.time()
                    })

            elif "bytes" in message:
                # Handle audio data
                raw_bytes = message["bytes"]
                audio_chunk = decode_audio_bytes(raw_bytes)

                if audio_chunk is None or len(audio_chunk) == 0:
                    continue

                # Process with Silero VAD
                vad_event = vad.process_chunk(audio_chunk)

                if vad_event.event_type == "speech_start":
                    await websocket.send_json({
                        "type": "vad_event",
                        "event": "speech_start",
                        "confidence": round(vad_event.confidence, 3),
                        "timestamp": vad_event.timestamp
                    })

                elif vad_event.event_type == "speech_end":
                    await websocket.send_json({
                        "type": "vad_event",
                        "event": "speech_end",
                        "confidence": round(vad_event.confidence, 3),
                        "timestamp": vad_event.timestamp
                    })

                    # Process the speech segment in background to keep main loop alive
                    if vad_event.audio_segment:
                        task = asyncio.create_task(
                            _process_speech_segment_with_signal(
                                websocket,
                                vad_event.audio_segment,
                                whisper_service,
                                ollama_service,
                                state,
                                vad,
                                processing_lock,
                                speaker_detector,
                                role_detector,
                                registry,
                                diarizer,
                                separator,
                                rag_service,
                                voice_embedder,
                            )
                        )
                        background_tasks.add(task)
                        task.add_done_callback(background_tasks.discard)

    except WebSocketDisconnect:
        print("WebSocket disconnected: audio-stream")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "timestamp": time.time()
            })
        except:
            pass
    finally:
        sweep_task.cancel()
        try:
            await sweep_task
        except (asyncio.CancelledError, Exception):
            pass
        vad.reset_states()
        print("WebSocket cleanup complete")


async def _get_suggestions(rag_service, ollama_service, text: str, role: str) -> list[str]:
    """Doctor-facing follow-up questions for a turn (RAG first, Ollama fallback)."""
    try:
        suggestions = await asyncio.wait_for(
            rag_service.get_suggested_questions_async(text, top_k=3),
            timeout=30.0,
        )
        if suggestions:
            print(f"🔍 RAG Suggestions ({role} spoke): {suggestions}")
            return suggestions
        print(f"📌 No RAG results, generating with Ollama (speaker={role})...")
        return await _generate_fallback_suggestions(ollama_service, text, role)
    except asyncio.TimeoutError:
        logger.warning("RAG timed out; generating fallback suggestions")
        return await _generate_fallback_suggestions(ollama_service, text, role)
    except Exception as e:
        logger.warning(f"RAG error: {e}; generating fallback suggestions")
        return await _generate_fallback_suggestions(ollama_service, text, role)


async def _transcribe_and_label_clip(
    clip: np.ndarray,
    sr: int,
    overlap_flag: bool,
    whisper_service,
    speaker_detector,
    role_detector,
    registry: "SpeakerRegistry",
    diar_start: float,
    diar_end: float,
    voice_embedder=None,
) -> Optional[dict]:
    """Transcribe one (possibly un-mixed) clip and produce a labeled turn dict."""
    from app.services.whisper_service import is_whisper_hallucination

    clip_dur_s = len(clip) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(np.square(clip)))) if len(clip) else 0.0
    text, lang = await asyncio.wait_for(
        whisper_service.transcribe_array(clip, sr), timeout=120.0
    )
    # Multi-voice diagnostics: show every clip Whisper sees, its loudness,
    # and which filter (if any) rejected it. If ChatGPT's voice is captured
    # but missing from the UI, the reject reason here tells you why.
    raw_preview = (text or "").strip().replace("\n", " ")[:100]
    print(
        f"🎧 Clip @{diar_start:.2f}-{diar_end:.2f}s "
        f"(dur={clip_dur_s:.2f}s, rms={rms:.4f}, overlap={overlap_flag}) "
        f"→ Whisper: '{raw_preview}'"
    )

    if not text.strip():
        print(f"  ❌ Dropped: empty transcript (audio probably too quiet — "
              f"check that ChatGPT's voice is loud enough to be captured)")
        return None
    if _is_garbage_transcript(text):
        print(f"  ❌ Dropped: garbage transcript filter")
        return None
    if is_whisper_hallucination(text):
        print(f"  ❌ Dropped: Whisper hallucination filter")
        return None
    # Belt-and-braces guard for YouTube-trained Whisper hallucinations
    # (see HALLUCINATION_SUBSTRINGS at the top of this module). Catches
    # variants the structured `is_whisper_hallucination` check might miss
    # (e.g. "Thanks for watching everyone!", trailing/leading filler text).
    lowered = text.lower()
    for hp in HALLUCINATION_SUBSTRINGS:
        if hp in lowered:
            print(f"  ❌ Dropped: hallucination substring '{hp}' in transcript")
            return None
    # NOTE: no per-clip RMS gate. The segment-level SEGMENT_PEAK_NOISE_FLOOR
    # check has already rejected pure noise; here we trust diarization plus
    # the content filters above so genuine quiet utterances (e.g. the second
    # voice in a multi-voice scenario) reach the UI.

    # Identity: prefer neural ECAPA-TDNN fingerprints (accurate per-voice, incl.
    # separated overlapping streams); fall back to lightweight MFCC embedding.
    if voice_embedder is not None and voice_embedder.available:
        emb = await asyncio.to_thread(voice_embedder.embed, clip, sr)
        speaker_id, is_new = registry.identify(emb, threshold=voice_embedder.recommended_threshold)
    else:
        profile = speaker_detector.extract_features_from_array(clip, sr)
        speaker_id, is_new = registry.identify(profile.embedding)
    content_role, content_conf = role_detector.detect_role(text)
    role, role_conf = registry.assign_role(speaker_id, content_role, content_conf)

    display_text = f"[overlapping speech] {text}" if overlap_flag else text
    print(
        f"  ✅ Kept as {speaker_id}/{role} ({role_conf:.0f}% conf, "
        f"is_new={is_new}, lang={lang})"
    )
    return {
        "speaker_id": speaker_id,
        "role": role,
        "role_confidence": round(role_conf, 1),
        "text": display_text,
        "start": round(diar_start, 2),
        "end": round(diar_end, 2),
        "is_new_speaker": is_new,
        "overlap": overlap_flag,
        "language": lang,
    }


async def _process_speech_segment(
    websocket: WebSocket,
    audio_segment_bytes: bytes,
    whisper_service,
    ollama_service,
    state: ConversationState,
    vad: VoiceActivityDetector,
    speaker_detector,
    role_detector,
    registry: "SpeakerRegistry",
    diarizer,
    separator,
    rag_service,
    voice_embedder=None,
):
    """Multi-speaker pipeline: diarize -> (un-mix overlap) -> transcribe per turn
    -> identify speaker -> assign role -> emit labeled turns -> AI reply on the
    last Patient turn."""
    try:
        if not await _safe_send_json(websocket, {
            "type": "processing",
            "stage": "transcribing",
            "message": "Identifying speakers...",
            "timestamp": time.time(),
        }):
            return

        sr = 16000
        audio = np.frombuffer(audio_segment_bytes, dtype=np.float32)
        seg_dur = len(audio) / sr if sr else 0.0
        seg_rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        seg_peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        print(
            f"📥 Speech segment received: dur={seg_dur:.2f}s, "
            f"rms={seg_rms:.4f}, peak={seg_peak:.4f}"
        )

        # Segment-level noise floor (see SEGMENT_PEAK_NOISE_FLOOR above for
        # rationale and calibration notes). Below the floor there's no audible
        # signal — only mic self-noise — so Whisper would hallucinate.
        if seg_peak < SEGMENT_PEAK_NOISE_FLOOR:
            print(
                f"🚫 Segment dropped: peak={seg_peak:.4f} < {SEGMENT_PEAK_NOISE_FLOOR} "
                f"(below mic noise floor — no signal)"
            )
            await _safe_send_json(websocket, {
                "type": "processing",
                "stage": "skipped",
                "message": "Audio too quiet, skipped",
                "timestamp": time.time(),
            })
            return

        if len(audio) < MIN_SEGMENT_SAMPLES:  # < 0.1s at 16 kHz
            await _safe_send_json(websocket, {
                "type": "processing",
                "stage": "skipped",
                "message": "Audio too short, skipped",
                "timestamp": time.time(),
            })
            return

        settings = get_settings()
        min_turn = settings.min_turn_duration_s

        # 0) PRE-DIARIZATION OVERLAP CHECK: if heavy talk-over is detected on the
        # whole segment, un-mix into separate voice streams BEFORE diarization.
        # This catches the case where two speakers are simultaneous for most of
        # the segment — diarization would otherwise pick one dominant voice and
        # silently drop the other.
        emitted_turns: list[dict] = []
        last_patient_text: Optional[str] = None
        last_patient_speaker_id: Optional[str] = None
        last_lang = "en"

        whole_sep = await asyncio.to_thread(separator.maybe_separate, audio, sr)
        if whole_sep.separated and whole_sep.streams:
            print(
                f"🔀 Pre-diarize: un-mixed full segment into {len(whole_sep.streams)} "
                f"stream(s) (overlap ratio={whole_sep.overlap_ratio:.2f})"
            )
            for stream in whole_sep.streams:
                turn = await _transcribe_and_label_clip(
                    stream, sr, True, whisper_service, speaker_detector,
                    role_detector, registry, 0.0, len(stream) / sr, voice_embedder,
                )
                if turn:
                    emitted_turns.append(turn)
                    last_lang = turn["language"]
                    if turn["role"] == "Patient":
                        last_patient_text = turn["text"].replace("[overlapping speech] ", "")
                        last_patient_speaker_id = turn["speaker_id"]

            # When the whole segment was un-mixed, skip per-turn diarization —
            # the streams are already speaker-separated.
            diar_turns = []
        else:
            # 1) DIARIZE the segment into per-speaker turns.
            diar_turns = await asyncio.to_thread(diarizer.diarize, audio, sr)
            print(f"🗣️ Diarization produced {len(diar_turns)} turn(s) via {diarizer.backend}")

        # 2) Process each diarized turn in temporal order.
        for dt in diar_turns:
            s_idx = max(0, int(dt.start * sr))
            e_idx = min(len(audio), int(dt.end * sr))
            clip = audio[s_idx:e_idx]
            if len(clip) < int(min_turn * sr):
                continue

            # 2a) Overlap handling: detect, and un-mix with Sepformer if available.
            sep = await asyncio.to_thread(separator.maybe_separate, clip, sr)

            if sep.separated and sep.streams:
                # True talk-over un-mixed into multiple voice streams.
                for stream in sep.streams:
                    turn = await _transcribe_and_label_clip(
                        stream, sr, False, whisper_service, speaker_detector,
                        role_detector, registry, dt.start, dt.end, voice_embedder,
                    )
                    if turn:
                        emitted_turns.append(turn)
                        last_lang = turn["language"]
                        if turn["role"] == "Patient":
                            last_patient_text = turn["text"].replace("[overlapping speech] ", "")
                            last_patient_speaker_id = turn["speaker_id"]
            else:
                # Single dominant voice (optionally flagged as overlapped).
                overlap_flag = sep.method == "flagged"
                turn = await _transcribe_and_label_clip(
                    clip, sr, overlap_flag, whisper_service, speaker_detector,
                    role_detector, registry, dt.start, dt.end, voice_embedder,
                )
                if turn:
                    emitted_turns.append(turn)
                    last_lang = turn["language"]
                    if turn["role"] == "Patient":
                        last_patient_text = turn["text"].replace("[overlapping speech] ", "")
                        last_patient_speaker_id = turn["speaker_id"]

        if not emitted_turns:
            await _safe_send_json(websocket, {
                "type": "transcript",
                "text": "[No valid speech]",
                "language": last_lang,
                "is_valid": False,
                "reason": "empty",
                "timestamp": time.time(),
            })
            return

        # 2a) Deduplicate redundant turns. We catch THREE cases:
        #
        #   (1) Pyannote emitted OVERLAPPING diarization turns in the same
        #       segment — e.g. A "I am not growing so tell me how to grow"
        #       AND B "so tell me how to grow". B is a substring of A.
        #   (2) Whisper transcribed the same audio TWICE with a tiny
        #       variation — e.g. "how to grown my height tell you about
        #       so so so so" vs "how to grow my height tell you about so
        #       so so so". Different by one character, but the same
        #       utterance. Caught by fuzzy similarity (≥ 0.85).
        #   (3) Cross-segment: the same phrase echoes from a still-open
        #       bubble in a previous segment (e.g. acoustic echo back into
        #       the mic). Caught by comparing against `state.open_turns`.
        FUZZY_DUP_THRESHOLD = 0.85

        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").replace("[overlapping speech] ", "")).strip().lower()

        def _is_dup(a: str, b: str) -> bool:
            if not a or not b:
                return False
            # Substring covers the "A contains B" diarization case.
            if a in b or b in a:
                return True
            # Fuzzy similarity covers the Whisper-retranscription case.
            return SequenceMatcher(None, a, b).ratio() >= FUZZY_DUP_THRESHOLD

        # Pre-compute normalized texts of currently-open bubbles for the
        # cross-segment check. These represent bubbles already on the UI.
        open_norms = [_norm(ot.text) for ot in state.open_turns.values()]

        deduped_turns: list[dict] = []
        for t in emitted_turns:
            t_norm = _norm(t["text"])
            if not t_norm:
                continue
            redundant = False

            # (1)+(2) in-segment: drop t if another turn in this segment is
            # longer AND duplicates t (substring OR fuzzy match).
            for u in emitted_turns:
                if u is t:
                    continue
                u_norm = _norm(u["text"])
                if len(u_norm) > len(t_norm) and _is_dup(t_norm, u_norm):
                    redundant = True
                    break

            # (3) cross-segment: drop t if it duplicates any currently-open
            # bubble on the UI.
            if not redundant:
                for o_norm in open_norms:
                    if _is_dup(t_norm, o_norm):
                        redundant = True
                        break

            if not redundant:
                deduped_turns.append(t)

        if len(deduped_turns) != len(emitted_turns):
            print(
                f"🧹 Deduplicated redundant turns: "
                f"{len(emitted_turns)} → {len(deduped_turns)}"
            )
            emitted_turns = deduped_turns
            # Rebuild last_patient_* from the deduped list so the AI driver
            # turn still points at a kept bubble.
            last_patient_text = None
            last_patient_speaker_id = None
            for t in emitted_turns:
                if t["role"] == "Patient":
                    last_patient_text = t["text"].replace("[overlapping speech] ", "")
                    last_patient_speaker_id = t["speaker_id"]

        # If dedup wiped out every turn (entire segment was a duplicate of
        # something already on screen), bail cleanly instead of falling
        # through to the AI reply step with empty turns.
        if not emitted_turns:
            print("🧹 All turns in this segment were duplicates of existing bubbles — nothing to emit")
            return

        # 2b) Merge consecutive turns from same speaker into one bubble.
        # Relaxed thresholds: a single continuous utterance from one person
        # frequently has brief mid-thought pauses and produces 60–70% role
        # confidence on each piece. The earlier strict gates (≥75% conf,
        # <0.5s gap) caused one spoken sentence to render as two bubbles.
        # The same_speaker_id + same_role check is still the safety net —
        # we never merge across different voices.
        MAX_GAP_S = 2.0
        MIN_CONF = 50.0
        merged_turns: list[dict] = []
        for turn in emitted_turns:
            if merged_turns:
                prev = merged_turns[-1]
                gap = turn["start"] - prev["end"]
                same_speaker = prev["speaker_id"] == turn["speaker_id"]
                same_role = prev["role"] == turn["role"]
                high_conf = (
                    prev["role_confidence"] >= MIN_CONF
                    and turn["role_confidence"] >= MIN_CONF
                )
                no_overlap = not turn.get("overlap", False) and not prev.get("overlap", False)
                tight_gap = gap < MAX_GAP_S

                if same_speaker and same_role and high_conf and no_overlap and tight_gap:
                    # Safe to merge
                    prev_text = prev["text"].rstrip(" .,!?")
                    new_text = turn["text"].lstrip()
                    if prev_text.endswith((".", "!", "?")):
                        prev["text"] = f"{prev_text} {new_text}"
                    else:
                        prev["text"] = f"{prev_text}. {new_text}"
                    prev["end"] = turn["end"]
                    prev["role_confidence"] = max(prev["role_confidence"], turn["role_confidence"])
                    continue
            merged_turns.append(dict(turn))

        if len(merged_turns) != len(emitted_turns):
            print(f"🔗 Merged {len(emitted_turns)} → {len(merged_turns)} bubble(s) (strict same-speaker rules)")

        # 3a) Route each merged turn through ConversationState so continuous
        # same-speaker utterances coalesce into ONE bubble across segments. The
        # state tracks per-speaker open bubbles so an interruption by speaker B
        # doesn't break speaker A's open bubble — A's resume continues it.
        now = time.time()
        coalesce_events: list[dict] = []
        for t in merged_turns:
            for ev in state.coalesce_or_open(t, now=now):
                coalesce_events.append(ev)
                await _safe_send_json(websocket, ev)

        # Attach the resolved turn_id back onto each merged turn for the legacy
        # message, so any client that reads `speaker_turns` can still correlate.
        new_or_update_by_speaker: dict[str, str] = {}
        for ev in coalesce_events:
            if ev["type"] in ("turn_new", "turn_update"):
                new_or_update_by_speaker[ev["speaker_id"]] = ev["turn_id"]
        for t in merged_turns:
            tid = new_or_update_by_speaker.get(t["speaker_id"])
            if tid:
                t["turn_id"] = tid

        # 3b) Legacy emit (kept for backwards-compat with older clients).
        await _safe_send_json(websocket, {
            "type": "speaker_turns",
            "turns": merged_turns,
            "language": last_lang,
            "timestamp": time.time(),
        })

        # 4) Drive the AI reply from the LAST PATIENT turn (per design decision).
        driver_text = last_patient_text or emitted_turns[-1]["text"].replace(
            "[overlapping speech] ", "")
        driver_role = "Patient" if last_patient_text else emitted_turns[-1]["role"]
        driver_speaker_id = last_patient_speaker_id or emitted_turns[-1]["speaker_id"]

        rag_suggestions = await _get_suggestions(
            rag_service, ollama_service, driver_text, driver_role)

        if not await _safe_send_json(websocket, {
            "type": "processing",
            "stage": "thinking",
            "message": "AI is thinking...",
            "timestamp": time.time(),
        }):
            return

        # MULTI-VOICE CONTEXT: build a role-tagged transcript of this segment.
        # When the user is talking to another voice agent (ChatGPT voice), both
        # voices arrive in the same audio segment and are diarized as separate
        # turns. The LLM must see BOTH — using per-speaker history alone hides
        # the other voice and the assistant replies without context.
        segment_user_text = ConversationState.build_segment_user_text(merged_turns) or driver_text

        # Use the unified chronological history so the LLM sees EVERY speaker's
        # prior turns, not just the driver's. This is what makes the multi-voice
        # mode work: the "other" voice (e.g. ChatGPT) becomes part of context.
        history = state.get_unified_history()
        try:
            reply = await asyncio.wait_for(
                ollama_service.ask_generic_english(segment_user_text, history, last_lang),
                timeout=300.0,
            )
            print(f"✅ AI Reply (driver={driver_role}/{driver_speaker_id}, "
                  f"{len(merged_turns)} turn(s) in segment): {reply}")
        except asyncio.TimeoutError:
            logger.error("Ollama response generation timed out")
            await _safe_send_json(websocket, {
                "type": "error",
                "message": "AI response generation took too long, please try again",
                "timestamp": time.time(),
            })
            return

        # Append this segment's combined transcript + reply to the unified history
        # so the next segment's LLM call sees the full multi-voice conversation.
        state.append_unified_user(segment_user_text)
        state.append_unified_assistant(reply)

        await _safe_send_json(websocket, {
            "type": "response",
            "transcript": driver_text,
            "reply": reply,
            "language": last_lang,
            "speaker_role": driver_role,
            "speaker_id": driver_speaker_id,
            "rag_suggestions": rag_suggestions,
            "turns": emitted_turns,
            "timestamp": time.time(),
        })

    except asyncio.TimeoutError:
        await _safe_send_json(websocket, {
            "type": "error",
            "message": "Processing timed out",
            "timestamp": time.time(),
        })
    except Exception as e:
        logger.error(f"Error processing speech segment: {e}", exc_info=True)
        print(f"Error processing speech segment: {e}")
        await _safe_send_json(websocket, {
            "type": "error",
            "message": f"Processing error: {str(e)}",
            "timestamp": time.time(),
        })


