import asyncio
import logging
import tempfile
from pathlib import Path
from threading import Lock

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from langdetect import DetectorFactory, detect
import noisereduce as nr
import librosa

from app.core.config import Settings

DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

WEAK_PHRASES = {
    "so guys", "thank you", "thanks for watching", "what's up", "hello",
    "bye", "okay", "ok", "hmm", "um", "uh", "oh", "ah", "yeah", "yes", "no",
    "you", "i", "the", "a", "an",
}

LANG_CODE_MAP = {
    "zh-cn": "zh", "zh-tw": "zh",
}


class WhisperService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = Lock()

    def _get_model(self) -> WhisperModel:
        """Lazy-load faster-whisper model with double-checked locking."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = WhisperModel(
                        model_size_or_path=self.settings.whisper_model,
                        device=self.settings.whisper_device,
                        compute_type=self.settings.whisper_compute_type,
                    )
                    logger.info(
                        f"Loaded faster-whisper model: {self.settings.whisper_model} "
                        f"(device={self.settings.whisper_device}, compute={self.settings.whisper_compute_type})"
                    )
        return self._model

    def _load_and_prepare_audio(self, file_path: Path) -> np.ndarray:
        """Load audio with librosa and apply noise reduction."""
        try:
            # Load audio at 16kHz mono
            audio, sr = librosa.load(str(file_path), sr=16000, mono=True)

            # Apply noise reduction (using already-installed noisereduce library)
            # stationary=False for speech, prop_decrease=0.75 for aggressive but natural reduction
            audio_reduced = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.75)

            logger.info(f"Loaded and processed audio: {len(audio_reduced)} samples at {sr}Hz")
            return audio_reduced
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise

    def _transcribe_with_options(
        self, model: WhisperModel, file_path: Path, language: str | None = None
    ) -> tuple[str, str]:
        """Transcribe using faster-whisper with optimized settings."""
        try:
            segments_generator, info = model.transcribe(
                str(file_path),
                task=self.settings.whisper_task,
                language=language,
                beam_size=5,  # Slightly larger for accuracy, still fast
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,  # Built-in VAD pre-filter for silence skipping
                vad_parameters={"min_silence_duration_ms": 300},  # Skip short silence periods
            )

            # Assemble full text from segments
            text = " ".join(segment.text.strip() for segment in segments_generator)
            text = text.strip()

            whisper_lang = info.language if info.language else "en"

            logger.info(
                f"Transcription result: lang={whisper_lang}, text_len={len(text)}, "
                f"language_prob={info.language_probability:.2f}"
            )

            return text, whisper_lang
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise

    def _is_weak_transcript(self, text: str) -> bool:
        """Check if transcript is too weak to process."""
        normalized = text.strip().lower().rstrip(".,!?")
        return len(normalized) < 4 or normalized in WEAK_PHRASES

    def _consensus_language(self, whisper_lang: str, text: str) -> str:
        """Determine the language using script detection + Whisper consensus."""
        import re

        # Priority 1: Check for any Indian script characters (most reliable)
        if re.search(r'[ऀ-ॿ]', text):  # Hindi/Devanagari
            logger.info("Detected Hindi script, language=hi")
            return "hi"
        if re.search(r'[ఀ-౿]', text):  # Telugu
            logger.info("Detected Telugu script, language=te")
            return "te"
        if re.search(r'[஀-௿]', text):  # Tamil
            logger.info("Detected Tamil script, language=ta")
            return "ta"
        if re.search(r'[઀-૿]', text):  # Gujarati
            logger.info("Detected Gujarati script, language=gu")
            return "gu"
        if re.search(r'[਀-੿]', text):  # Punjabi
            logger.info("Detected Punjabi script, language=pa")
            return "pa"
        if re.search(r'[぀-ゟ゠-ヿ一-鿿]', text):  # Japanese/Chinese
            return whisper_lang if whisper_lang in ["ja", "zh"] else "en"

        # Priority 2: Trust Whisper if it detects Indian language
        indian_languages = {"hi", "te", "ta", "kn", "ml", "mr", "bn", "ur", "gu", "pa", "or", "as"}
        if whisper_lang in indian_languages:
            logger.info("Whisper detected Indian language: %s", whisper_lang)
            return whisper_lang

        # Priority 3: Default to English for mixed/uncertain cases
        logger.info("Defaulting to English (mixed/uncertain language)")
        return "en"

    def _transcribe_sync(self, file_path: Path) -> tuple[str, str]:
        """Synchronous transcription with retry logic."""
        model = self._get_model()

        # Load and preprocess audio
        audio = self._load_and_prepare_audio(file_path)

        # Save to temporary WAV file for faster-whisper
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = Path(tmp_wav.name)
        tmp_wav.close()

        try:
            # Write audio to temp WAV
            sf.write(str(tmp_path), audio, 16000)

            # Transcribe with options
            text, whisper_lang = self._transcribe_with_options(model, tmp_path)

            # Retry logic for weak transcripts
            if self._is_weak_transcript(text):
                logger.info("Weak transcript detected, retrying with language hint: %s", whisper_lang)
                retry_text, retry_lang = self._transcribe_with_options(model, tmp_path, language=whisper_lang)
                if len(retry_text) > len(text):
                    text = retry_text
                    whisper_lang = retry_lang

            # Determine final language
            language = self._consensus_language(whisper_lang, text)

            logger.info(
                f"WHISPER RESULT | whisper_lang={whisper_lang} | consensus_lang={language} | "
                f"text_len={len(text)} | text={text[:100]}"
            )

            return text, language
        finally:
            tmp_path.unlink(missing_ok=True)

    async def transcribe_audio(self, file_path: Path) -> tuple[str, str]:
        """Async wrapper for transcription."""
        return await asyncio.to_thread(self._transcribe_sync, file_path)

    def _transcribe_array_sync(self, audio: np.ndarray, sr: int) -> tuple[str, str]:
        """Transcribe an in-memory float32 audio slice (per-speaker sub-segment)."""
        model = self._get_model()
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Apply the same noise reduction used for file-based transcription.
        try:
            audio = nr.reduce_noise(y=audio, sr=16000, stationary=False, prop_decrease=0.75)
        except Exception as e:
            logger.warning(f"Noise reduction skipped for array transcription: {e}")

        segments_generator, info = model.transcribe(
            audio.astype(np.float32),
            task=self.settings.whisper_task,
            beam_size=5,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(segment.text.strip() for segment in segments_generator).strip()
        whisper_lang = info.language if info.language else "en"
        language = self._consensus_language(whisper_lang, text)
        return text, language

    async def transcribe_array(self, audio: np.ndarray, sr: int = 16000) -> tuple[str, str]:
        """Async wrapper for in-memory array transcription."""
        return await asyncio.to_thread(self._transcribe_array_sync, audio, sr)

    async def translate_to_english(self, file_path: Path) -> str:
        """Transcribe and translate to English."""
        text, _ = await self.transcribe_audio(file_path)
        return text
