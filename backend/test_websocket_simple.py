#!/usr/bin/env python3
"""
Simplified WebSocket test - focus on continuous loop, not transcription accuracy.
Tests that after one speech segment is processed, the system stays listening.
"""

import asyncio
import json
import websockets
import numpy as np
import time


async def test_continuous_listening():
    """Test that system continues listening after processing"""
    uri = "ws://localhost:8000/api/v1/ws/audio-stream"

    print("🔌 Testing continuous hands-free listening mode...")
    print(f"   Connecting to: {uri}\n")

    try:
        async with websockets.connect(uri) as ws:
            # Track state
            events_received = {
                "speech_start": 0,
                "speech_end": 0,
                "response": 0,
                "ready_for_next": 0,
                "error": 0,
            }

            # Start session
            await ws.send(json.dumps({"type": "start_session"}))
            print("✅ Session started")

            # Receive and count messages for 45 seconds
            start_time = time.time()
            print("\n📡 Listening for messages (45 seconds)...\n")

            try:
                while time.time() - start_time < 45:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(msg)
                        msg_type = data.get("type")

                        if msg_type == "vad_event":
                            event = data.get("event")
                            if event in events_received:
                                events_received[event] += 1
                                conf = data.get("confidence", 0)
                                print(f"  📊 VAD {event}: confidence={conf:.2f}")

                        elif msg_type == "response":
                            events_received["response"] += 1
                            reply = data.get("reply", "")[:50]
                            print(f"  ✅ Response received: {reply}...")

                        elif msg_type == "ready_for_next":
                            events_received["ready_for_next"] += 1
                            print(f"  ⏩ Ready for next speech")

                        elif msg_type == "error":
                            events_received["error"] += 1
                            err = data.get("message")
                            print(f"  ❌ Error: {err}")

                    except asyncio.TimeoutError:
                        pass

            except Exception as e:
                print(f"Stopped: {e}")

            # End session
            await ws.send(json.dumps({"type": "end_session"}))

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # Print results
    print(f"\n📋 Results after 45 seconds:")
    print(f"  Speech starts detected: {events_received['speech_start']}")
    print(f"  Speech ends detected: {events_received['speech_end']}")
    print(f"  Responses received: {events_received['response']}")
    print(f"  Ready-for-next signals: {events_received['ready_for_next']}")
    print(f"  Errors: {events_received['error']}")

    # Check if continuous loop is working
    print(f"\n✨ Analysis:")
    if events_received["speech_end"] > 0 and events_received["response"] > 0:
        print(f"  ✅ System detected speech and generated responses")
    if events_received["ready_for_next"] > 0:
        print(f"  ✅ Backend signaled ready-for-next (continuous mode active)")
    if events_received["error"] > 0:
        print(f"  ⚠️  Errors occurred - check logs")

    success = (
        events_received["response"] > 0
        and events_received["ready_for_next"] > 0
    )
    if success:
        print(f"\n✅ CONTINUOUS MODE WORKING!")
    else:
        print(f"\n⚠️  Issues detected - review output above")


if __name__ == "__main__":
    asyncio.run(test_continuous_listening())
