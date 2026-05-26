#!/usr/bin/env python3
"""
Test script for continuous hands-free WebSocket mode.
Simulates audio capture and verifies the loop continues properly after each response.
"""

import asyncio
import json
import websockets
import numpy as np
import time
from typing import Optional


class MockAudioGenerator:
    """Generate mock PCM16 audio data at 16kHz"""

    def __init__(self):
        self.sample_rate = 16000

    def generate_silence(self, duration_ms: int) -> bytes:
        """Generate silence as PCM16"""
        samples = int(self.sample_rate * duration_ms / 1000)
        audio = np.zeros(samples, dtype=np.int16)
        return audio.tobytes()

    def generate_tone(self, duration_ms: int, frequency: int = 440) -> bytes:
        """Generate a tone at given frequency"""
        samples = int(self.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, samples, False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def generate_speech_like(self, duration_ms: int) -> bytes:
        """Generate speech-like audio (mix of frequencies)"""
        samples = int(self.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, samples, False)
        # Mix of frequencies to simulate speech
        audio = (
            0.2 * np.sin(2 * np.pi * 200 * t)
            + 0.2 * np.sin(2 * np.pi * 400 * t)
            + 0.2 * np.sin(2 * np.pi * 800 * t)
        )
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()


async def send_continuous_audio(
    ws,
    audio_gen: MockAudioGenerator,
    num_iterations: int = 3,
    verbose: bool = True,
):
    """
    Simulate continuous audio input with multiple speech segments.
    This tests the loop: capture -> process -> continue listening -> repeat.
    """
    print(f"\n🎤 Starting {num_iterations} continuous speech segments...")

    for iteration in range(num_iterations):
        print(f"\n─── Iteration {iteration + 1}/{num_iterations} ───")

        # Phase 1: Silence
        silence = audio_gen.generate_silence(500)
        chunk_size = 1024
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i : i + chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.01)
        print(f"  🔇 Sent silence (500ms)")

        # Phase 2: Speech-like audio
        speech = audio_gen.generate_speech_like(2000)
        for i in range(0, len(speech), chunk_size):
            chunk = speech[i : i + chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.01)
        print(f"  🔊 Sent speech (2000ms)")

        # Phase 3: More silence (should trigger speech_end)
        silence = audio_gen.generate_silence(1500)
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i : i + chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.01)
        print(f"  🔇 Sent trailing silence (1500ms) - should trigger speech_end")

        # Wait for processing (3-5 seconds for Whisper + Ollama)
        print(f"  ⏳ Waiting for backend processing...")
        await asyncio.sleep(6)


async def receive_messages(ws, timeout: int = 30) -> Optional[list]:
    """Receive all messages from WebSocket within timeout"""
    messages = []
    start_time = time.time()

    try:
        while time.time() - start_time < timeout:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(msg)
                messages.append(data)

                msg_type = data.get("type", "unknown")
                if msg_type == "vad_event":
                    event = data.get("event")
                    confidence = data.get("confidence", 0)
                    print(f"  📊 VAD Event: {event} (confidence: {confidence:.2f})")
                elif msg_type == "transcript":
                    text = data.get("text", "")
                    valid = data.get("is_valid", False)
                    status = "✅" if valid else "❌"
                    print(f"  {status} Transcript: {text[:60]}...")
                elif msg_type == "response":
                    reply = data.get("reply", "")
                    print(f"  🤖 AI Response: {reply[:60]}...")
                elif msg_type == "ready_for_next":
                    print(f"  ✅ Backend signaled ready for next speech")
                elif msg_type == "processing":
                    stage = data.get("stage", "")
                    print(f"  ⏳ Processing: {stage}")
                elif msg_type == "error":
                    error_msg = data.get("message", "")
                    print(f"  ❌ Error: {error_msg}")
                elif msg_type == "keepalive":
                    pass
                else:
                    print(f"  ℹ️  {msg_type}: {data}")

            except asyncio.TimeoutError:
                continue

    except Exception as e:
        print(f"  ❌ Error receiving: {e}")

    return messages


async def test_continuous_hands_free():
    """Main test: continuous hands-free WebSocket loop"""
    uri = "ws://localhost:8000/api/v1/ws/audio-stream"
    audio_gen = MockAudioGenerator()

    try:
        print(f"🔌 Connecting to WebSocket: {uri}")
        async with websockets.connect(uri) as ws:
            print(f"✅ Connected!")

            # Start message receiver in background
            receiver_task = asyncio.create_task(receive_messages(ws, timeout=45))

            # Wait for initial connection message
            await asyncio.sleep(1)

            # Send start_session command
            print(f"📤 Sending start_session command")
            await ws.send(json.dumps({"type": "start_session"}))
            await asyncio.sleep(1)

            # Send continuous audio with multiple iterations
            await send_continuous_audio(ws, audio_gen, num_iterations=3)

            # Stop session
            print(f"\n📤 Sending end_session command")
            await ws.send(json.dumps({"type": "end_session"}))
            await asyncio.sleep(2)

            # Wait for receiver to finish
            messages = await receiver_task
            print(f"\n📊 Total messages received: {len(messages)}")

            # Analyze results
            print(f"\n📋 Summary:")
            vad_starts = sum(1 for m in messages if m.get("type") == "vad_event" and m.get("event") == "speech_start")
            vad_ends = sum(1 for m in messages if m.get("type") == "vad_event" and m.get("event") == "speech_end")
            transcripts = sum(1 for m in messages if m.get("type") == "transcript" and m.get("is_valid"))
            responses = sum(1 for m in messages if m.get("type") == "response")
            ready_signals = sum(1 for m in messages if m.get("type") == "ready_for_next")

            print(f"  📊 Speech starts detected: {vad_starts}")
            print(f"  📊 Speech ends detected: {vad_ends}")
            print(f"  ✅ Valid transcripts: {transcripts}")
            print(f"  🤖 AI responses: {responses}")
            print(f"  ✅ Ready-for-next signals: {ready_signals}")

            # Validation
            print(f"\n✨ Validation:")
            if vad_starts > 0:
                print(f"  ✅ VAD detected speech starts")
            else:
                print(f"  ❌ VAD did not detect any speech starts")

            if vad_ends > 0:
                print(f"  ✅ VAD detected speech ends")
            else:
                print(f"  ❌ VAD did not detect any speech ends")

            if responses > 0:
                print(f"  ✅ Backend processed responses")
            else:
                print(f"  ❌ No responses received")

            if ready_signals > 0:
                print(f"  ✅ Backend sent ready-for-next signals (continuous mode working!)")
            else:
                print(f"  ⚠️  No ready-for-next signals (may indicate pipeline issue)")

            if responses == 3:
                print(f"\n✅ SUCCESS: All 3 iterations processed!")
            else:
                print(f"\n⚠️  Only {responses}/3 iterations processed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_continuous_hands_free())
