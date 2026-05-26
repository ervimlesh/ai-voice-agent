from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencies import get_voice_agent_service
from app.controllers.agent_controller import AgentController
from app.schemas.chat import AgentResponse, SpeakerDetectionResponse, TextAskRequest
from app.services.voice_agent_service import VoiceAgentService

router = APIRouter(prefix="/agent", tags=["agent"])


def get_agent_controller(
    service: VoiceAgentService = Depends(get_voice_agent_service),
) -> AgentController:
    return AgentController(service)


@router.post("/text", response_model=AgentResponse)
async def ask_by_text(
    payload: TextAskRequest,
    controller: AgentController = Depends(get_agent_controller),
):
    return await controller.ask_by_text(payload.message, payload.history)


@router.post("/voice", response_model=AgentResponse)
async def ask_by_voice(
    audio: UploadFile = File(...),
    history: str = Form(default="[]"),
    controller: AgentController = Depends(get_agent_controller),
):
    return await controller.ask_by_voice(audio, history)


@router.post("/voice-with-speaker-detection", response_model=SpeakerDetectionResponse)
async def ask_by_voice_with_speaker_detection(
    audio: UploadFile = File(...),
    history: str = Form(default="[]"),
    controller: AgentController = Depends(get_agent_controller),
):
    return await controller.ask_by_voice_with_speaker_detection(audio, history)


@router.post("/reset-speaker-profiles")
async def reset_speaker_profiles(
    controller: AgentController = Depends(get_agent_controller),
):
    return controller.reset_speaker_profiles()


@router.get("/speaker-info")
async def get_speaker_info(
    controller: AgentController = Depends(get_agent_controller),
):
    return controller.get_speaker_info()


@router.post("/start-hands-free-session")
async def start_hands_free_session(
    controller: AgentController = Depends(get_agent_controller),
):
    return controller.start_hands_free_session()


@router.post("/end-hands-free-session")
async def end_hands_free_session(
    controller: AgentController = Depends(get_agent_controller),
):
    return controller.end_hands_free_session()


@router.get("/session-info")
async def get_session_info(
    controller: AgentController = Depends(get_agent_controller),
):
    return controller.get_session_info()


@router.post("/detect-voice-activity")
async def detect_voice_activity(
    audio: UploadFile = File(...),
    controller: AgentController = Depends(get_agent_controller),
):
    return await controller.detect_voice_activity(audio)


@router.post("/check-silence")
async def check_silence(
    audio: UploadFile = File(...),
    silence_threshold: float = 2.0,
    controller: AgentController = Depends(get_agent_controller),
):
    return await controller.check_silence(audio, silence_threshold)
