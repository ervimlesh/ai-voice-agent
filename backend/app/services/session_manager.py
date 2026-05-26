import uuid
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from app.schemas.chat import ChatMessage
from app.models.speaker_profile import SpeakerProfile


@dataclass
class ConversationSession:
    """Manages a continuous conversation session"""
    session_id: str
    created_at: datetime
    updated_at: datetime
    history: List[ChatMessage] = field(default_factory=list)
    speaker_profiles: Dict[str, SpeakerProfile] = field(default_factory=dict)
    doctor_profile: Optional[SpeakerProfile] = None
    patient_profile: Optional[SpeakerProfile] = None
    current_speaker_id: Optional[str] = None
    is_active: bool = True
    turn_count: int = 0
    
    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation history"""
        self.history.append(ChatMessage(role=role, content=content))
        self.updated_at = datetime.now()
    
    def update_speaker_profile(self, speaker_id: str, profile: SpeakerProfile) -> None:
        """Update speaker profile"""
        self.speaker_profiles[speaker_id] = profile
        self.updated_at = datetime.now()
    
    def get_history(self) -> List[ChatMessage]:
        """Get conversation history"""
        return self.history
    
    def to_dict(self) -> dict:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "history_length": len(self.history),
            "turn_count": self.turn_count,
            "is_active": self.is_active,
            "current_speaker": self.current_speaker_id,
            "has_doctor": self.doctor_profile is not None,
            "has_patient": self.patient_profile is not None,
        }


class SessionManager:
    """Manages multiple conversation sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.active_session_id: Optional[str] = None
    
    def create_session(self) -> str:
        """Create new conversation session"""
        session_id = str(uuid.uuid4())
        session = ConversationSession(
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.sessions[session_id] = session
        self.active_session_id = session_id
        print(f"📊 New session created: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def get_active_session(self) -> Optional[ConversationSession]:
        """Get currently active session"""
        if self.active_session_id:
            return self.sessions.get(self.active_session_id)
        return None
    
    def set_active_session(self, session_id: str) -> bool:
        """Set active session"""
        if session_id in self.sessions:
            self.active_session_id = session_id
            return True
        return False
    
    def end_session(self, session_id: str) -> bool:
        """End a session"""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            if self.active_session_id == session_id:
                self.active_session_id = None
            print(f"🏁 Session ended: {session_id}")
            return True
        return False
    
    def get_all_sessions(self) -> List[dict]:
        """Get all sessions info"""
        return [session.to_dict() for session in self.sessions.values()]
    
    def clear_old_sessions(self, max_sessions: int = 10) -> None:
        """Keep only recent sessions"""
        if len(self.sessions) > max_sessions:
            # Sort by updated_at and keep only recent ones
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].updated_at,
                reverse=True
            )
            self.sessions = dict(sorted_sessions[:max_sessions])
            print(f"🧹 Cleaned up old sessions, keeping {max_sessions}")
