#!/usr/bin/env python3
"""
Test script for speaker detection functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.speaker_detector_service import SpeakerDetector
from app.services.voice_agent_service import VoiceAgentService
from app.services.whisper_service import WhisperService
from app.services.ollama_service import OllamaService
from app.schemas.chat import ChatMessage
from app.core.config import get_settings


async def test_speaker_detector():
    """Test the speaker detector with sample audio files"""
    print("🔬 Testing Speaker Detection System")
    print("=" * 50)
    
    # Initialize services
    settings = get_settings()
    whisper_service = WhisperService(settings)
    ollama_service = OllamaService(settings)
    voice_agent_service = VoiceAgentService(whisper_service, ollama_service)
    speaker_detector = SpeakerDetector()
    
    print("✅ Services initialized successfully")
    
    # Test speaker info
    speaker_info = voice_agent_service.get_speaker_info()
    print(f"📊 Initial speaker info: {speaker_info}")
    
    # Check for test audio files
    test_audio_dir = Path("test_audio")
    if not test_audio_dir.exists():
        print(f"⚠️ No test_audio directory found at {test_audio_dir}")
        print("Creating test_audio directory - please add audio files named:")
        print("  - doctor1.wav")
        print("  - doctor2.wav") 
        print("  - patient1.wav")
        print("  - patient2.wav")
        test_audio_dir.mkdir(exist_ok=True)
        return
    
    audio_files = list(test_audio_dir.glob("*.wav")) + list(test_audio_dir.glob("*.mp3"))
    if not audio_files:
        print("⚠️ No audio files found in test_audio directory")
        return
    
    print(f"🎵 Found {len(audio_files)} audio files for testing")
    
    # Test feature extraction
    for audio_file in audio_files:
        print(f"\n🔍 Processing: {audio_file.name}")
        try:
            profile = speaker_detector.extract_features(str(audio_file))
            print(f"   Pitch: {profile.pitch:.1f} Hz")
            print(f"   Spectral Centroid: {profile.spectral_centroid:.1f} Hz")
            print(f"   MFCC: {len(profile.mfcc)} coefficients")
            print(f"   Zero Crossing Rate: {profile.zero_crossing_rate:.4f}")
            print(f"   RMS Energy: {profile.rms_energy:.4f}")
            print(f"   Speech Rate: {profile.speech_rate:.1f} wpm")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Test speaker comparison if we have multiple files
    if len(audio_files) >= 2:
        print(f"\n🔗 Testing speaker comparison")
        profiles = []
        for audio_file in audio_files[:2]:  # Test first two files
            try:
                profile = speaker_detector.extract_features(str(audio_file))
                profiles.append((audio_file.name, profile))
            except Exception as e:
                print(f"   ❌ Error extracting from {audio_file.name}: {str(e)}")
        
        if len(profiles) >= 2:
            name1, profile1 = profiles[0]
            name2, profile2 = profiles[1]
            similarity = speaker_detector.calculate_similarity(profile1, profile2)
            print(f"   Similarity between {name1} and {name2}: {similarity:.1f}%")
            
            if similarity > 95:
                print("   👥 Likely the same speaker")
            elif similarity > 85:
                print("   ❓ Uncertain - could be same speaker")
            else:
                print("   👥 Different speakers")
    
    print("\n✅ Speaker detection test completed!")


async def test_voice_agent_with_detection():
    """Test the voice agent service with speaker detection"""
    print("\n🤖 Testing Voice Agent with Speaker Detection")
    print("=" * 50)
    
    settings = get_settings()
    whisper_service = WhisperService(settings)
    ollama_service = OllamaService(settings)
    voice_agent_service = VoiceAgentService(whisper_service, ollama_service)
    
    # Reset speaker profiles
    voice_agent_service.reset_speaker_profiles()
    
    # Check for test audio files
    test_audio_dir = Path("test_audio")
    audio_files = list(test_audio_dir.glob("*.wav")) + list(test_audio_dir.glob("*.mp3"))
    
    if len(audio_files) < 2:
        print("⚠️ Need at least 2 audio files for testing speaker detection")
        return
    
    history = []
    
    # Process first audio file
    print(f"\n🎤 Processing first audio: {audio_files[0].name}")
    try:
        response = await voice_agent_service.ask_by_audio_with_speaker_detection(
            str(audio_files[0]), history
        )
        print(f"   Transcript: {response.transcript}")
        print(f"   Reply: {response.reply}")
        print(f"   Speaker ID: {response.speaker_info.speaker_id}")
        print(f"   Confidence: {response.speaker_info.confidence:.1f}%")
        print(f"   New Speaker: {response.speaker_info.is_new_speaker}")
        history = response.history
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return
    
    # Process second audio file
    print(f"\n🎤 Processing second audio: {audio_files[1].name}")
    try:
        response = await voice_agent_service.ask_by_audio_with_speaker_detection(
            str(audio_files[1]), history
        )
        print(f"   Transcript: {response.transcript}")
        print(f"   Reply: {response.reply}")
        print(f"   Speaker ID: {response.speaker_info.speaker_id}")
        print(f"   Confidence: {response.speaker_info.confidence:.1f}%")
        print(f"   New Speaker: {response.speaker_info.is_new_speaker}")
        history = response.history
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return
    
    # Show final speaker info
    speaker_info = voice_agent_service.get_speaker_info()
    print(f"\n📊 Final speaker info:")
    print(f"   Total speakers: {speaker_info['total_speakers']}")
    print(f"   Has doctor profile: {speaker_info['has_doctor_profile']}")
    print(f"   Has patient profile: {speaker_info['has_patient_profile']}")
    
    print("\n✅ Voice agent test completed!")


if __name__ == "__main__":
    print("🚀 Starting Speaker Detection Tests")
    
    try:
        asyncio.run(test_speaker_detector())
        asyncio.run(test_voice_agent_with_detection())
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
