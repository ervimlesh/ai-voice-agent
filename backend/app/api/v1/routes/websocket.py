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
from app.core.config import get_settings
from app.schemas.chat import ChatMessage

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
    history: list,
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
            history,
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

    history: list[ChatMessage] = []
    session_active = True
    processing_lock = asyncio.Lock()
    background_tasks = set()

    # Per-conversation dynamic speaker registry (handles 3+ speakers).
    registry = SpeakerRegistry(get_settings())

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
                        history.clear()
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
                        history.clear()
                        vad.reset_states()
                        registry.reset()  # Reset speaker identities for new conversation
                        await websocket.send_json({
                            "type": "history_reset",
                            "timestamp": time.time()
                        })

                    elif cmd_type == "set_history":
                        # Allow client to sync history
                        raw_history = cmd.get("history", [])
                        history = [ChatMessage(**m) for m in raw_history]
                        await websocket.send_json({
                            "type": "history_synced",
                            "history_length": len(history),
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
                                history,
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
    text, lang = await asyncio.wait_for(
        whisper_service.transcribe_array(clip, sr), timeout=120.0
    )
    if _is_garbage_transcript(text) or not text.strip():
        return None

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
    history: list,
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

        if len(audio) < 1600:  # < 0.1s
            await _safe_send_json(websocket, {
                "type": "processing",
                "stage": "skipped",
                "message": "Audio too short, skipped",
                "timestamp": time.time(),
            })
            return

        settings = get_settings()
        min_turn = settings.min_turn_duration_s

        # 1) DIARIZE the segment into per-speaker turns (runs off the event loop).
        diar_turns = await asyncio.to_thread(diarizer.diarize, audio, sr)
        print(f"🗣️ Diarization produced {len(diar_turns)} turn(s) via {diarizer.backend}")

        emitted_turns: list[dict] = []
        last_patient_text: Optional[str] = None
        last_lang = "en"

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

        # 3) Emit all labeled turns at once.
        await _safe_send_json(websocket, {
            "type": "speaker_turns",
            "turns": emitted_turns,
            "language": last_lang,
            "timestamp": time.time(),
        })

        # 4) Drive the AI reply from the LAST PATIENT turn (per design decision).
        driver_text = last_patient_text or emitted_turns[-1]["text"].replace(
            "[overlapping speech] ", "")
        driver_role = "Patient" if last_patient_text else emitted_turns[-1]["role"]

        rag_suggestions = await _get_suggestions(
            rag_service, ollama_service, driver_text, driver_role)

        if not await _safe_send_json(websocket, {
            "type": "processing",
            "stage": "thinking",
            "message": "AI is thinking...",
            "timestamp": time.time(),
        }):
            return

        try:
            reply = await asyncio.wait_for(
                ollama_service.ask_generic_english(driver_text, history, last_lang),
                timeout=300.0,
            )
            print(f"✅ AI Reply (driver={driver_role}): {reply}")
        except asyncio.TimeoutError:
            logger.error("Ollama response generation timed out")
            await _safe_send_json(websocket, {
                "type": "error",
                "message": "AI response generation took too long, please try again",
                "timestamp": time.time(),
            })
            return

        history.append(ChatMessage(role="user", content=driver_text))
        history.append(ChatMessage(role="assistant", content=reply))
        if len(history) > 20:
            history[:] = history[-20:]

        await _safe_send_json(websocket, {
            "type": "response",
            "transcript": driver_text,
            "reply": reply,
            "language": last_lang,
            "speaker_role": driver_role,
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


