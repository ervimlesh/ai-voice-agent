#!/usr/bin/env python3
"""
Quick demo of the speaker detection system
"""

import requests
import json
import time
import os

API_BASE = "http://localhost:8000/api/v1"

def demo_speaker_detection():
    """Demonstrate the speaker detection API"""
    print("🎤 Speaker Detection System Demo")
    print("=" * 40)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE}/health")
        print("✅ Server is running")
    except:
        print("❌ Server not found. Please start the backend server:")
        print("   cd backend && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        return
    
    # Reset speaker profiles
    print("\n🔄 Resetting speaker profiles...")
    response = requests.post(f"{API_BASE}/agent/reset-speaker-profiles")
    print(f"   {response.json()['message']}")
    
    # Get initial speaker info
    response = requests.get(f"{API_BASE}/agent/speaker-info")
    print(f"📊 Initial speakers: {response.json()['total_speakers']}")
    
    # Check for test audio files
    test_files = ["test_audio/doctor1.wav", "test_audio/patient1.wav"]
    if not all(os.path.exists(f) for f in test_files):
        print("❌ Test audio files not found. Run test_speaker_detection.py first.")
        return
    
    # Process first audio (doctor)
    print("\n🎤 Processing doctor audio...")
    with open("test_audio/doctor1.wav", "rb") as f:
        files = {"audio": ("doctor1.wav", f, "audio/wav")}
        data = {"history": "[]"}
        response = requests.post(f"{API_BASE}/agent/voice-with-speaker-detection", 
                               files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Transcript: '{result['transcript']}'")
        print(f"   Speaker ID: {result['speaker_info']['speaker_id']}")
        print(f"   Confidence: {result['speaker_info']['confidence']:.1f}%")
        print(f"   New Speaker: {result['speaker_info']['is_new_speaker']}")
        print(f"   Response preview: {result['reply'][:100]}...")
    else:
        print(f"   ❌ Error: {response.status_code}")
        return
    
    # Process second audio (patient)
    print("\n🎤 Processing patient audio...")
    with open("test_audio/patient1.wav", "rb") as f:
        files = {"audio": ("patient1.wav", f, "audio/wav")}
        data = {"history": json.dumps([{"role": "user", "content": "you"}, 
                                     {"role": "assistant", "content": "Hello!"}])}
        response = requests.post(f"{API_BASE}/agent/voice-with-speaker-detection", 
                               files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Transcript: '{result['transcript']}'")
        print(f"   Speaker ID: {result['speaker_info']['speaker_id']}")
        print(f"   Confidence: {result['speaker_info']['confidence']:.1f}%")
        print(f"   New Speaker: {result['speaker_info']['is_new_speaker']}")
        print(f"   Response preview: {result['reply'][:100]}...")
    else:
        print(f"   ❌ Error: {response.status_code}")
        return
    
    # Get final speaker info
    response = requests.get(f"{API_BASE}/agent/speaker-info")
    info = response.json()
    print(f"\n📊 Final speaker info:")
    print(f"   Total speakers: {info['total_speakers']}")
    print(f"   Has doctor profile: {info['has_doctor_profile']}")
    print(f"   Has patient profile: {info['has_patient_profile']}")
    
    print("\n✅ Demo completed successfully!")
    print("\n🎯 Key Features Demonstrated:")
    print("   ✅ Automatic speaker detection")
    print("   ✅ Doctor/Patient role assignment")
    print("   ✅ Context-aware responses")
    print("   ✅ Speaker profile tracking")

if __name__ == "__main__":
    demo_speaker_detection()
