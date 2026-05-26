from fastapi import APIRouter

from app.api.v1.routes.agent import router as agent_router
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.websocket import router as ws_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(agent_router)
api_router.include_router(ws_router)
