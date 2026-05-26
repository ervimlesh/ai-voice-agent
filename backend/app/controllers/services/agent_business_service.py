from pathlib import Path

from app.schemas.chat import AgentResponse, ChatMessage, SpeakerDetectionResponse
from app.services.voice_agent_service import VoiceAgentService


class AgentBusinessService:
    def __init__(self, voice_agent_service: VoiceAgentService):
        self.voice_agent_service = voice_agent_service

    async def process_text_query(
        self, message: str, history: list
    ) -> AgentResponse:
        """Process text-based query"""
        return await self.voice_agent_service.ask_by_text(message, history)

    async def process_voice_query(
        self, audio_path: Path, history: list
    ) -> AgentResponse:
        """Process voice query and return response"""
        return await self.voice_agent_service.ask_by_audio(audio_path, history)

    async def process_voice_with_speaker_detection(
        self, audio_path: Path, history: list
    ) -> SpeakerDetectionResponse:
        """Process voice with speaker detection"""
        return await self.voice_agent_service.ask_by_audio_with_speaker_detection(
            audio_path, history
        )

    def reset_speaker_profiles(self) -> dict:
        """Reset all speaker profiles"""
        self.voice_agent_service.reset_speaker_profiles()
        return {"message": "Speaker profiles reset successfully"}

    def get_speaker_info(self) -> dict:
        """Get speaker information and profiles"""
        return self.voice_agent_service.get_speaker_info()

    def start_continuous_session(self) -> dict:
        """Start hands-free continuous session"""
        session_id = self.voice_agent_service.start_continuous_session()
        return {
            "session_id": session_id,
            "status": "active",
            "message": "Hands-free session started. Doctor can now speak continuously without clicking.",
        }

    def end_continuous_session(self) -> dict:
        """End current hands-free session"""
        success = self.voice_agent_service.end_continuous_session()
        return {
            "success": success,
            "message": "Hands-free session ended"
            if success
            else "No active session",
        }

    def get_session_info(self) -> dict:
        """Get current session information"""
        return self.voice_agent_service.get_session_info()

    def detect_voice_activity(self, audio_path: str) -> dict:
        """Detect if audio contains speech"""
        has_speech, confidence = self.voice_agent_service.detect_voice_activity(
            audio_path
        )
        return {
            "has_speech": has_speech,
            "confidence": confidence,
            "should_continue": has_speech and confidence > 0.5,
        }

    def check_silence(self, audio_path: str, silence_threshold: float) -> dict:
        """Check if audio has sufficient silence"""
        should_stop = self.voice_agent_service.should_stop_recording(
            audio_path, silence_threshold
        )
        return {
            "should_stop": should_stop,
            "silence_threshold": silence_threshold,
            "message": "Silence detected - ready to process"
            if should_stop
            else "Still recording...",
        }
