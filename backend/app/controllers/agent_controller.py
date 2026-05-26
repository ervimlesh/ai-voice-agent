from fastapi import UploadFile

from app.controllers.services.agent_business_service import AgentBusinessService
from app.controllers.services.audio_service import AudioService
from app.schemas.chat import AgentResponse, SpeakerDetectionResponse
from app.services.voice_agent_service import VoiceAgentService


class AgentController:
    def __init__(self, voice_agent_service: VoiceAgentService):
        self.audio_service = AudioService(voice_agent_service)
        self.business_service = AgentBusinessService(voice_agent_service)

    async def ask_by_text(self, message: str, history: list) -> AgentResponse:
        return await self.business_service.process_text_query(message, history)

    async def ask_by_voice(
        self, audio: UploadFile, history: str
    ) -> AgentResponse:
        self.audio_service.validate_audio_file(audio)
        parsed_history = self.audio_service.parse_chat_history(history)
        audio_path = await self.audio_service.save_audio_file(audio)

        try:
            return await self.business_service.process_voice_query(
                audio_path, parsed_history
            )
        finally:
            self.audio_service.cleanup_temp_file(audio_path)

    async def ask_by_voice_with_speaker_detection(
        self, audio: UploadFile, history: str
    ) -> SpeakerDetectionResponse:
        self.audio_service.validate_audio_file(audio)
        parsed_history = self.audio_service.parse_chat_history(history)
        audio_path = await self.audio_service.save_audio_file(audio)

        try:
            return await self.business_service.process_voice_with_speaker_detection(
                audio_path, parsed_history
            )
        finally:
            self.audio_service.cleanup_temp_file(audio_path)

    def reset_speaker_profiles(self) -> dict:
        return self.business_service.reset_speaker_profiles()

    def get_speaker_info(self) -> dict:
        return self.business_service.get_speaker_info()

    def start_hands_free_session(self) -> dict:
        return self.business_service.start_continuous_session()

    def end_hands_free_session(self) -> dict:
        return self.business_service.end_continuous_session()

    def get_session_info(self) -> dict:
        return self.business_service.get_session_info()

    async def detect_voice_activity(self, audio: UploadFile) -> dict:
        audio_path = await self.audio_service.save_audio_file(audio)
        try:
            return self.business_service.detect_voice_activity(str(audio_path))
        finally:
            self.audio_service.cleanup_temp_file(audio_path)

    async def check_silence(
        self, audio: UploadFile, silence_threshold: float = 2.0
    ) -> dict:
        audio_path = await self.audio_service.save_audio_file(audio)
        try:
            return self.business_service.check_silence(str(audio_path), silence_threshold)
        finally:
            self.audio_service.cleanup_temp_file(audio_path)
