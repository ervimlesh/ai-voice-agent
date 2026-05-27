import numpy as np
import torch
import time
import io
import logging
import threading
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Shared singleton for the Silero VAD model ──
_vad_model = None
_vad_model_lock = threading.Lock()


def _get_shared_model():
    """Load and cache the Silero VAD model (thread-safe singleton)."""
    global _vad_model
    if _vad_model is None:
        with _vad_model_lock:
            if _vad_model is None:
                print("⏳ Loading Silero VAD model (one-time)...")
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
                model.eval()
                _vad_model = model
                print("✅ Silero VAD model loaded successfully")
    return _vad_model


@dataclass
class VADEvent:
    """Represents a VAD event"""
    event_type: str  # "speech_start", "speech_end", "speech_active", "silence"
    timestamp: float
    confidence: float = 0.0
    audio_segment: Optional[bytes] = None


class VoiceActivityDetector:
    """Silero VAD-based voice activity detector for real-time audio processing.
    
    Each instance maintains its own state but shares the underlying model.
    Create one instance per WebSocket session.
    """
    
    def __init__(self):
        self.sample_rate = 16000
        self.window_size = 512  # Silero VAD requires 512 samples at 16kHz
        self._model = _get_shared_model()
        
        # State tracking
        self.speech_active = False
        self.silence_start_time: Optional[float] = None
        self.speech_start_time: Optional[float] = None
        self.audio_buffer: List[np.ndarray] = []
        
        # Thresholds — tuned for multi-voice capture. Lower speech threshold so
        # quieter speaker-played audio (e.g. ChatGPT voice through the laptop
        # speakers, which is acoustically attenuated by the time it reaches the
        # mic) still triggers a segment instead of being mistaken for silence.
        self.speech_threshold = 0.35
        self.min_speech_duration_ms = 400
        # Longer trailing-silence window: only cut a segment after a real pause
        # (~1 second). Shorter pauses within a sentence won't fragment the speech.
        self.min_silence_duration_ms = 1000
        self.speech_pad_ms = 200
    
    def reset_states(self):
        """Reset all VAD states for a new session"""
        if self._model is not None:
            self._model.reset_states()
        self.speech_active = False
        self.silence_start_time = None
        self.speech_start_time = None
        self.audio_buffer.clear()
        print("VAD states reset")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> VADEvent:
        """
        Process a single audio chunk with Silero VAD.
        audio_chunk: numpy array of float32 audio samples at 16kHz
        Returns: VADEvent indicating what happened
        """
        try:
            # Ensure correct shape and type
            if len(audio_chunk) == 0:
                return VADEvent(event_type="silence", timestamp=time.time())
            
            audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32))
            
            # Process in windows of 512 samples
            confidence = 0.0
            num_windows = max(1, len(audio_tensor) // self.window_size)
            
            for i in range(num_windows):
                start = i * self.window_size
                end = start + self.window_size
                if end > len(audio_tensor):
                    break
                window = audio_tensor[start:end]
                speech_prob = self._model(window, self.sample_rate).item()
                confidence = max(confidence, speech_prob)
            
            now = time.time()
            is_speech = confidence >= self.speech_threshold
            
            # State machine for speech detection
            if is_speech and not self.speech_active:
                # Speech just started
                self.speech_active = True
                self.speech_start_time = now
                self.silence_start_time = None
                self.audio_buffer.clear()
                self.audio_buffer.append(audio_chunk)
                return VADEvent(
                    event_type="speech_start",
                    timestamp=now,
                    confidence=confidence
                )
            
            elif is_speech and self.speech_active:
                # Speech continues
                self.audio_buffer.append(audio_chunk)
                self.silence_start_time = None
                return VADEvent(
                    event_type="speech_active",
                    timestamp=now,
                    confidence=confidence
                )
            
            elif not is_speech and self.speech_active:
                # Potential speech end - track silence duration
                self.audio_buffer.append(audio_chunk)
                
                if self.silence_start_time is None:
                    self.silence_start_time = now
                
                silence_duration_ms = (now - self.silence_start_time) * 1000
                
                if silence_duration_ms >= self.min_silence_duration_ms:
                    # Enough silence to consider speech ended
                    speech_duration_ms = 0
                    if self.speech_start_time:
                        speech_duration_ms = (now - self.speech_start_time) * 1000
                    
                    if speech_duration_ms >= self.min_speech_duration_ms:
                        # Valid speech segment ended
                        self.speech_active = False
                        segment = self._get_buffered_audio()
                        self.audio_buffer.clear()
                        self.silence_start_time = None
                        self.speech_start_time = None
                        
                        segment_bytes = segment.tobytes() if segment is not None else None
                        
                        return VADEvent(
                            event_type="speech_end",
                            timestamp=now,
                            confidence=confidence,
                            audio_segment=segment_bytes
                        )
                    else:
                        # Speech was too short, discard
                        self.speech_active = False
                        self.audio_buffer.clear()
                        self.silence_start_time = None
                        self.speech_start_time = None
                        return VADEvent(
                            event_type="silence",
                            timestamp=now,
                            confidence=confidence
                        )
                else:
                    # Brief pause, might resume speaking
                    return VADEvent(
                        event_type="speech_active",
                        timestamp=now,
                        confidence=confidence
                    )
            
            else:
                # Silence, no active speech
                return VADEvent(
                    event_type="silence",
                    timestamp=now,
                    confidence=confidence
                )
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            print(f"Error processing audio chunk: {e}")
            return VADEvent(event_type="silence", timestamp=time.time())
    
    def _get_buffered_audio(self) -> Optional[np.ndarray]:
        """Concatenate buffered audio into a single segment"""
        if not self.audio_buffer:
            return None
        return np.concatenate(self.audio_buffer)
    
    def detect_speech_segments(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if audio contains speech (batch mode - compatible with old interface)
        Returns: (has_speech, confidence)
        """
        try:
            if len(audio_data) < self.window_size:
                return False, 0.0
            
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            
            max_confidence = 0.0
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_tensor) - self.window_size + 1, self.window_size):
                window = audio_tensor[i:i + self.window_size]
                speech_prob = self._model(window, self.sample_rate).item()
                max_confidence = max(max_confidence, speech_prob)
                total_frames += 1
                if speech_prob >= self.speech_threshold:
                    speech_frames += 1
            
            # Reset model states after batch processing
            self._model.reset_states()
            
            has_speech = speech_frames > 0
            avg_confidence = speech_frames / total_frames if total_frames > 0 else 0.0
            
            return has_speech, float(max(max_confidence, avg_confidence))
            
        except Exception as e:
            print(f"Error detecting speech: {str(e)}")
            return False, 0.0
    
    def should_stop_recording(self, audio_data: np.ndarray, silence_threshold: float = 2.0) -> bool:
        """
        Determine if recording should stop based on trailing silence.
        silence_threshold: seconds of silence to trigger stop
        """
        try:
            if len(audio_data) < self.window_size:
                return False
            
            audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            
            # Check the last N seconds of audio for silence
            check_samples = int(silence_threshold * self.sample_rate)
            if len(audio_tensor) < check_samples:
                return False
            
            tail = audio_tensor[-check_samples:]
            
            speech_detected = False
            for i in range(0, len(tail) - self.window_size + 1, self.window_size):
                window = tail[i:i + self.window_size]
                speech_prob = self._model(window, self.sample_rate).item()
                if speech_prob >= self.speech_threshold:
                    speech_detected = True
                    break
            
            self._model.reset_states()
            return not speech_detected
            
        except Exception as e:
            print(f"Error checking stop condition: {str(e)}")
            return False
