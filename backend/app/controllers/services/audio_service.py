import json
import tempfile
from pathlib import Path

import aiofiles
from fastapi import HTTPException, UploadFile, status

from app.schemas.chat import ChatMessage
from app.services.voice_agent_service import VoiceAgentService

class AudioService:
    def __init__(self, voice_agent_service: VoiceAgentService):
        self.voice_agent_service = voice_agent_service

    async def save_audio_file(self, audio: UploadFile) -> Path:
        """Save uploaded audio file to temporary location"""
        suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)

        async with aiofiles.open(temp_path, "wb") as out_file:
            while chunk := await audio.read(1024 * 1024):
                await out_file.write(chunk)

        return temp_path

    def validate_audio_file(self, audio: UploadFile) -> None:
        """Validate audio file type"""
        if not audio.content_type or not audio.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please upload a valid audio file.",
            )

    def parse_chat_history(self, history_str: str) -> list:
        """Parse and validate chat history JSON"""
        try:
            return [ChatMessage(**item) for item in json.loads(history_str)]
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid chat history payload.",
            ) from exc

    def cleanup_temp_file(self, file_path: Path) -> None:
        """Remove temporary file"""
        file_path.unlink(missing_ok=True)
