from app.schemas.chat import AgentResponse, ChatMessage, SpeakerDetectionResponse, SpeakerInfo
from app.services.ollama_service import OllamaService
from app.services.whisper_service import WhisperService
from app.services.speaker_detector_service import SpeakerDetector
from app.services.voice_activity_detector import VoiceActivityDetector
from app.services.session_manager import SessionManager, ConversationSession
from app.models.speaker_profile import SpeakerProfile
from typing import List, Tuple, Optional
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


class VoiceAgentService:
    def __init__(self, whisper_service: WhisperService, ollama_service: OllamaService):
        self.whisper_service = whisper_service
        self.ollama_service = ollama_service
        self.speaker_detector = SpeakerDetector()
        self.voice_activity_detector = VoiceActivityDetector()
        self.session_manager = SessionManager()
        self.speaker_profiles: List[Tuple[str, SpeakerProfile]] = []  # (speaker_id, profile)
        self.current_speaker_id: Optional[str] = None
        self.speaker_counter = 0
        self.doctor_profile: Optional[SpeakerProfile] = None
        self.patient_profile: Optional[SpeakerProfile] = None
        self.current_session: Optional[ConversationSession] = None

    async def ask_by_text(self, message: str, history: list[ChatMessage]) -> AgentResponse:
        # Detect language of text input (for logging/awareness)
        try:
            detected_language = detect(message)
        except:
            detected_language = "en"

        print(f"📝 Text detected language: {detected_language}")
        print(f"📝 User message: {message}")

        # Always respond in English, regardless of input language
        # Pass detected language for better translation handling
        reply = await self.ollama_service.ask_generic_english(message, history, detected_language)
        print(f"💬 Reply (English): {reply}")

        updated_history = [*history, ChatMessage(role="user", content=message), ChatMessage(role="assistant", content=reply)]
        return AgentResponse(transcript=message, reply=reply, history=updated_history)

    async def ask_by_audio(self, audio_path, history: list[ChatMessage]) -> AgentResponse:
        transcript, detected_language = await self.whisper_service.transcribe_audio(audio_path)
        print(f"🎤 Detected language: {detected_language}")
        print(f"📝 Transcript (in {detected_language}): {transcript}")

        # Handle empty transcript
        if not transcript or transcript.strip() == "":
            transcript = "[No speech detected]"
            reply = "I didn't hear any speech. Please try speaking more clearly or check your microphone."
        else:
            # Always respond in English, regardless of input language
            # Pass detected language for better translation handling
            reply = await self.ollama_service.ask_generic_english(transcript, history, detected_language)
            print(f"💬 Reply (English): {reply}")

        updated_history = [*history, ChatMessage(role="user", content=transcript), ChatMessage(role="assistant", content=reply)]
        return AgentResponse(transcript=transcript, reply=reply, history=updated_history)

    async def ask_by_audio_with_speaker_detection(self, audio_path: str, history: list[ChatMessage]) -> SpeakerDetectionResponse:
        """Process audio with automatic speaker detection and role switching"""
        # Transcribe audio
        transcript, detected_language = await self.whisper_service.transcribe_audio(audio_path)
        print(f"🎤 Detected language: {detected_language}")
        print(f"📝 Transcript: {transcript}")
        
        # Handle empty transcript
        if not transcript or transcript.strip() == "":
            transcript = "[No speech detected]"
            reply = "I didn't hear any speech. Please try speaking more clearly or check your microphone."
            speaker_info = SpeakerInfo(
                speaker_id="unknown",
                confidence=0.0,
                is_new_speaker=True
            )
            updated_history = [*history, ChatMessage(role="user", content=transcript), ChatMessage(role="assistant", content=reply)]
            return SpeakerDetectionResponse(
                transcript=transcript,
                reply=reply,
                history=updated_history,
                speaker_info=speaker_info
            )
        
        # Extract voice features and detect speaker
        current_profile = self.speaker_detector.extract_features(audio_path)
        
        # Detect speaker
        speaker_id, confidence = self.speaker_detector.detect_speaker(audio_path, self.speaker_profiles)
        is_new_speaker = False
        
        if speaker_id is None or confidence < 85.0:
            # New speaker detected
            self.speaker_counter += 1
            speaker_id = f"speaker_{self.speaker_counter}"
            self.speaker_profiles.append((speaker_id, current_profile))
            is_new_speaker = True
            print(f"🔊 New speaker detected: {speaker_id} (confidence: {confidence:.1f}%)")
        else:
            print(f"🔊 Known speaker detected: {speaker_id} (confidence: {confidence:.1f}%)")
        
        # Determine if this is doctor or patient based on first speaker
        if self.doctor_profile is None:
            # First speaker becomes the doctor
            self.doctor_profile = current_profile
            role_prefix = "Doctor"
            print(f"👨‍⚕️ First speaker identified as DOCTOR")
        elif self.patient_profile is None and not self._is_same_speaker(current_profile, self.doctor_profile):
            # Second different speaker becomes the patient
            self.patient_profile = current_profile
            role_prefix = "Patient"
            print(f"👨‍⚕️ Second different speaker identified as PATIENT")
        else:
            # Determine role based on profile comparison
            if self._is_same_speaker(current_profile, self.doctor_profile, 95.0):
                role_prefix = "Doctor"
            elif self.patient_profile and self._is_same_speaker(current_profile, self.patient_profile, 95.0):
                role_prefix = "Patient"
            else:
                # Uncertain, default to patient for safety
                role_prefix = "Patient"
                print(f"⚠️ Uncertain speaker identification, defaulting to Patient")
        
        # Get response from Ollama - always in English
        # Speaker detection info is returned with English response
        # Pass detected language for better handling
        reply = await self.ollama_service.ask_generic_english(transcript, history, detected_language)
        print(f"💬 Reply ({role_prefix} - English): {reply}")
        
        # Update history
        updated_history = [*history, ChatMessage(role="user", content=transcript), ChatMessage(role="assistant", content=reply)]
        
        # Create speaker info
        speaker_info = SpeakerInfo(
            speaker_id=speaker_id,
            confidence=confidence,
            is_new_speaker=is_new_speaker
        )
        
        return SpeakerDetectionResponse(
            transcript=transcript,
            reply=reply,
            history=updated_history,
            speaker_info=speaker_info
        )
    
    def _is_same_speaker(self, profile1: SpeakerProfile, profile2: SpeakerProfile, threshold: float = 95.0) -> bool:
        """Check if two profiles represent the same speaker"""
        if profile1 is None or profile2 is None:
            return False
        return self.speaker_detector.is_same_speaker(profile1, profile2, threshold)
    
    def _get_context_aware_prompt(self, transcript: str, role_prefix: str, is_new_speaker: bool) -> str:
        """Generate context-aware prompt based on speaker role"""
        if role_prefix == "Doctor":
            if is_new_speaker:
                return f"""A doctor is speaking for the first time: "{transcript}"
                
Respond as a helpful AI assistant to the doctor. Ask relevant follow-up questions or provide medical information as needed."""
            else:
                return f"""The doctor continues speaking: "{transcript}"
                
Respond as a helpful AI assistant. Continue the conversation with the doctor, providing relevant medical assistance."""
        else:  # Patient
            if is_new_speaker:
                return f"""A patient is speaking for the first time: "{transcript}"
                
Respond as a helpful AI assistant to the patient. Show empathy and ask about their symptoms or concerns."""
            else:
                return f"""The patient continues speaking: "{transcript}"
                
Respond as a helpful AI assistant. Continue the conversation with the patient, providing appropriate medical guidance."""
    
    def reset_speaker_profiles(self):
        """Reset all speaker profiles (for new conversation)"""
        self.speaker_profiles.clear()
        self.current_speaker_id = None
        self.speaker_counter = 0
        self.doctor_profile = None
        self.patient_profile = None
        print("🔄 Speaker profiles reset")
    
    def get_speaker_info(self) -> dict:
        """Get current speaker information"""
        return {
            "total_speakers": len(self.speaker_profiles),
            "current_speaker": self.current_speaker_id,
            "has_doctor_profile": self.doctor_profile is not None,
            "has_patient_profile": self.patient_profile is not None,
            "speaker_profiles": [
                {"id": sid, "profile": profile.to_dict()} 
                for sid, profile in self.speaker_profiles
            ]
        }
    
    def start_continuous_session(self) -> str:
        """Start a new continuous conversation session"""
        session_id = self.session_manager.create_session()
        self.current_session = self.session_manager.get_session(session_id)
        self.reset_speaker_profiles()
        print(f"🚀 Continuous session started: {session_id}")
        return session_id
    
    def end_continuous_session(self) -> bool:
        """End current continuous session"""
        if self.current_session:
            session_id = self.current_session.session_id
            self.session_manager.end_session(session_id)
            self.current_session = None
            print(f"🏁 Continuous session ended: {session_id}")
            return True
        return False
    
    def get_session_info(self) -> dict:
        """Get current session information"""
        if self.current_session:
            return {
                "session_id": self.current_session.session_id,
                "is_active": self.current_session.is_active,
                "turn_count": self.current_session.turn_count,
                "history_length": len(self.current_session.history),
                "created_at": self.current_session.created_at.isoformat(),
                "updated_at": self.current_session.updated_at.isoformat(),
            }
        return {"session_id": None, "is_active": False}
    
    def detect_voice_activity(self, audio_path: str) -> Tuple[bool, float]:
        """Detect if audio contains speech"""
        try:
            import soundfile as sf
            audio_data, sr = sf.read(audio_path)
            # Resample if needed
            if sr != self.voice_activity_detector.sample_rate:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.voice_activity_detector.sample_rate)
            
            has_speech, confidence = self.voice_activity_detector.detect_speech_segments(audio_data)
            print(f"🎙️ Voice activity: {has_speech} (confidence: {confidence:.2f})")
            return has_speech, confidence
        except Exception as e:
            print(f"Error detecting voice activity: {str(e)}")
            return False, 0.0
    
    def should_stop_recording(self, audio_path: str, silence_threshold: float = 2.0) -> bool:
        """Check if recording should stop based on silence detection"""
        try:
            import soundfile as sf
            audio_data, sr = sf.read(audio_path)
            # Resample if needed
            if sr != self.voice_activity_detector.sample_rate:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.voice_activity_detector.sample_rate)
            
            should_stop = self.voice_activity_detector.should_stop_recording(audio_data, silence_threshold)
            return should_stop
        except Exception as e:
            print(f"Error checking stop condition: {str(e)}")
            return False
