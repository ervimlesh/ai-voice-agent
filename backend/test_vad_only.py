#!/usr/bin/env python3
"""
Test VAD loop without Whisper - verify the continuous listening works.
"""

import sys
sys.path.insert(0, '/home/swathy/Downloads/ai-voice-agent-production/backend')

from app.services.voice_activity_detector import VoiceActivityDetector
import numpy as np
import time

def generate_speech_like(duration_ms: int) -> np.ndarray:
    """Generate speech-like audio"""
    sample_rate = 16000
    samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, samples, False)
    audio = (
        0.2 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.2 * np.sin(2 * np.pi * 800 * t)
    )
    return audio.astype(np.float32)

def test_vad_continuous():
    """Test VAD in continuous mode"""
    vad = VoiceActivityDetector()

    print("🎙️  Testing VAD continuous listening loop...")
    print("    Running 2 iterations of speech -> silence -> reset\n")

    for iteration in range(2):
        print(f"─── Iteration {iteration + 1} ───")

        # Phase 1: Silence
        silence = np.zeros(4000, dtype=np.float32)  # 250ms
        for i in range(0, len(silence), 512):
            chunk = silence[i:i+512]
            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512-len(chunk)))
            event = vad.process_chunk(chunk)
            if event.event_type != "silence":
                print(f"  Event: {event.event_type} (confidence: {event.confidence:.2f})")

        # Phase 2: Speech
        speech = generate_speech_like(1000)  # 1 second
        for i in range(0, len(speech), 512):
            chunk = speech[i:i+512]
            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512-len(chunk)))
            event = vad.process_chunk(chunk)
            if event.event_type != "silence":
                print(f"  Event: {event.event_type} (confidence: {event.confidence:.2f})")

        # Phase 3: Silence to trigger speech_end
        silence2 = np.zeros(8000, dtype=np.float32)  # 500ms
        for i in range(0, len(silence2), 512):
            chunk = silence2[i:i+512]
            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512-len(chunk)))
            event = vad.process_chunk(chunk)
            if event.event_type != "silence":
                print(f"  Event: {event.event_type} (confidence: {event.confidence:.2f})")
                if event.event_type == "speech_end" and event.audio_segment:
                    segment_len = len(np.frombuffer(event.audio_segment, dtype=np.float32))
                    print(f"    Audio segment length: {segment_len} samples (~{segment_len/16000*1000:.0f}ms)")

        # Reset for next iteration
        print(f"  Resetting VAD for next iteration...")
        vad.reset_states()
        print()

    print("✅ VAD continuous loop test complete!")

if __name__ == "__main__":
    test_vad_continuous()
