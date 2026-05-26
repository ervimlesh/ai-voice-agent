from app.core.config import get_settings


class HealthController:
    @staticmethod
    def health_check() -> dict:
        settings = get_settings()
        return {
            "status": "ok",
            "app": settings.app_name,
            "environment": settings.app_env,
            "whisper_model": settings.whisper_model,
            "ollama_model": settings.ollama_model,
        }
