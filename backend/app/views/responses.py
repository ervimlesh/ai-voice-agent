from typing import Any, Dict, Optional


class ResponseFormatter:
    @staticmethod
    def success(data: Any, message: Optional[str] = None) -> Dict:
        return {
            "status": "success",
            "data": data,
            "message": message,
        }

    @staticmethod
    def error(message: str, code: Optional[str] = None) -> Dict:
        return {
            "status": "error",
            "message": message,
            "code": code,
        }

    @staticmethod
    def paginated(
        data: Any, total: int, page: int, page_size: int
    ) -> Dict:
        return {
            "status": "success",
            "data": data,
            "pagination": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
            },
        }
