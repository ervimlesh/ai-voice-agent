from fastapi import APIRouter

from app.controllers.health_controller import HealthController

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    return HealthController.health_check()

