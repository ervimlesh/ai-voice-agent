import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload heavy ML models at startup so the first utterance isn't slow.

    Conquers the diarization/separation cold-start: the pyannote pipeline and
    Sepformer model (when available) are loaded here instead of lazily on the
    first speech segment.
    """
    try:
        from app.api.dependencies import (
            get_whisper_service,
            get_diarization_service,
            get_separation_service,
        )

        logger.info("⏳ Preloading models (whisper, diarization, separation)...")
        get_whisper_service()._get_model()
        get_diarization_service()
        get_separation_service()
        logger.info("✅ Model preload complete")
    except Exception as e:
        logger.warning(f"Model preload encountered an issue (continuing): {e}")
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Production-ready FastAPI backend for an all-language voice agent that replies only in English.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,         
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/")
async def root():
    return {
        "message": "AI Voice Agent API is running",
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
    }
