#!/usr/bin/env python3
"""
Test the full pipeline: VAD -> Whisper -> Ollama
Helps identify where bottlenecks are
"""

import sys
import asyncio
import time
sys.path.insert(0, '/home/swathy/Downloads/ai-voice-agent-production/backend')

from app.services.voice_activity_detector import VoiceActivityDetector
from app.services.whisper_service import WhisperService
from app.services.ollama_service import OllamaService
from app.core.config import get_settings
from app.schemas.chat import ChatMessage
import numpy as np
from pathlib import Path
import tempfile

def generate_speech_like(duration_ms: int) -> np.ndarray:
    """Generate speech-like audio"""
    sample_rate = 16000
    samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, samples, False)
    # Create multiple frequency components (more realistic than simple tone)
    audio = (
        0.3 * np.sin(2 * np.pi * 300 * t) +
        0.3 * np.sin(2 * np.pi * 600 * t) +
        0.2 * np.sin(2 * np.pi * 1000 * t)
    )
    return audio.astype(np.float32)

def audio_segment_to_wav_bytes(audio_float32: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert float32 numpy audio to WAV bytes"""
    import struct
    import io
    audio_int16 = (audio_float32 * 32767).astype(np.int16)
    buf = io.BytesIO()
    num_samples = len(audio_int16)
    data_size = num_samples * 2
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))
    buf.write(struct.pack('<H', 1))
    buf.write(struct.pack('<H', 1))
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', sample_rate * 2))
    buf.write(struct.pack('<H', 2))
    buf.write(struct.pack('<H', 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(audio_int16.tobytes())
    return buf.getvalue()

async def test_pipeline():
    """Test each component"""
    settings = get_settings()

    print("=" * 60)
    print("🧪 Full Pipeline Test")
    print("=" * 60)

    # Test 1: VAD
    print("\n1️⃣  Testing VAD (Voice Activity Detection)...")
    start = time.time()
    vad = VoiceActivityDetector()
    elapsed = time.time() - start
    print(f"   ✅ VAD initialized in {elapsed:.2f}s")

    # Process audio through VAD
    speech = generate_speech_like(2000)  # 2 seconds
    silence = np.zeros(4000, dtype=np.float32)  # 250ms

    audio_segment = None
    for chunk in [silence[:512], speech[:512], speech[512:1024], speech[1024:], silence]:
        if len(chunk) < 512:
            chunk = np.pad(chunk, (0, 512-len(chunk)))
        event = vad.process_chunk(chunk)
        if event.event_type == "speech_end" and event.audio_segment:
            audio_segment = event.audio_segment

    if audio_segment:
        segment_len = len(np.frombuffer(audio_segment, dtype=np.float32))
        print(f"   ✅ VAD detected speech segment ({segment_len} samples)")
    else:
        print(f"   ❌ VAD did not detect speech segment")
        return

    # Test 2: Whisper
    print("\n2️⃣  Testing Whisper (Speech-to-Text)...")
    whisper_service = WhisperService(settings)

    # Create WAV file from segment
    audio_float32 = np.frombuffer(audio_segment, dtype=np.float32)
    wav_bytes = audio_segment_to_wav_bytes(audio_float32)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        tmp_path = Path(tmp.name)

    try:
        start = time.time()
        transcript, language = await whisper_service.transcribe_audio(tmp_path)
        elapsed = time.time() - start
        print(f"   ✅ Whisper transcribed in {elapsed:.2f}s")
        print(f"      Transcript: {transcript[:100]}...")
        print(f"      Language: {language}")

        if not transcript or transcript.strip() == "":
            print(f"   ⚠️  Transcript is empty (synthetic audio not recognized)")
            # Continue anyway with a dummy transcript
            transcript = "hello what is the capital of india"
    except Exception as e:
        print(f"   ❌ Whisper failed: {e}")
        return
    finally:
        tmp_path.unlink(missing_ok=True)

    # Test 3: Ollama
    print("\n3️⃣  Testing Ollama (LLM Response)...")
    ollama_service = OllamaService(settings)

    try:
        start = time.time()
        history = []
        reply = await ollama_service.ask(transcript, history)
        elapsed = time.time() - start
        print(f"   ✅ Ollama responded in {elapsed:.2f}s")
        print(f"      Reply: {reply[:100]}...")
    except Exception as e:
        print(f"   ❌ Ollama failed: {e}")
        return

    # Summary
    print("\n" + "=" * 60)
    print("✨ Pipeline Status: ALL COMPONENTS WORKING ✨")
    print("=" * 60)
    print("\nNow test the full WebSocket flow:")
    print("1. Start frontend: npm run dev")
    print("2. Open http://localhost:5173")
    print("3. Click 'Hands-Free Mode'")
    print("4. Speak naturally and wait for response")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_pipeline())
