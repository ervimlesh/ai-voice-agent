from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant", "system"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class TextAskRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class AgentResponse(BaseModel):
    transcript: str
    reply: str
    history: list[ChatMessage]


class SpeakerInfo(BaseModel):
    speaker_id: str
    confidence: float
    is_new_speaker: bool


class SpeakerDetectionResponse(BaseModel):
    transcript: str
    reply: str
    history: list[ChatMessage]
    speaker_info: SpeakerInfo
