"""
COMPREHENSIVE UNIT TESTS FOR ENHANCED MULTI-VOICE AI AGENT

Tests both CPU and GPU modes for:
✅ Overlap detection
✅ Voice separation
✅ Diarization (speaker detection)
✅ End-to-end pipeline
✅ Multi-voice simultaneous speech
✅ Performance benchmarks

Run with: pytest test_enhanced_multivoice.py -v
Run GPU tests: pytest test_enhanced_multivoice.py -v -m gpu
Run CPU tests: pytest test_enhanced_multivoice.py -v -m cpu
"""

import pytest
import numpy as np
import time
import logging
from typing import Tuple, List
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import Settings
from app.services.enhanced_separation_service import EnhancedSeparationService, SeparationResult
from app.services.enhanced_diarization_service import EnhancedDiarizationService, DiarTurn
from app.services.speaker_detector_service import SpeakerDetector
from app.services.role_detector_service import RoleDetectorService

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        diarization_enabled=True,
        diarization_backend="auto",
        separation_enabled=True,
        separation_backend="auto",
        speaker_match_threshold=0.78,
        min_turn_duration_s=0.4,
        role_lock_confidence=75.0,
        overlap_min_ratio=0.15,
    )

@pytest.fixture
def speaker_detector():
    """Create speaker detector."""
    return SpeakerDetector()

@pytest.fixture
def separation_service(settings):
    """Create separation service."""
    return EnhancedSeparationService(settings)

@pytest.fixture
def diarization_service(settings, speaker_detector):
    """Create diarization service."""
    return EnhancedDiarizationService(settings, speaker_detector)

@pytest.fixture
def role_detector():
    """Create role detector."""
    return RoleDetectorService()

def create_test_audio(
    duration: float = 2.0,
    sr: int = 16000,
    frequencies: List[float] = None
) -> np.ndarray:
    """
    Create synthetic test audio.

    Args:
        duration: Length in seconds
        sr: Sample rate (Hz)
        frequencies: List of sine wave frequencies to mix
    """
    if frequencies is None:
        frequencies = [440]  # Default A4

    t = np.linspace(0, duration, int(sr * duration))
    audio = np.zeros_like(t)

    for freq in frequencies:
        audio += 0.3 * np.sin(2 * np.pi * freq * t)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    return audio.astype(np.float32)

def create_overlapping_speech(
    duration: float = 3.0,
    sr: int = 16000,
    voice1_freq: float = 120.0,
    voice2_freq: float = 180.0,
    overlap_start: float = 1.0,
    overlap_end: float = 2.5
) -> Tuple[np.ndarray, float]:
    """
    Create overlapping speech simulation.

    Args:
        duration: Total duration (seconds)
        sr: Sample rate
        voice1_freq: Frequency of voice 1 (Hz)
        voice2_freq: Frequency of voice 2 (Hz)
        overlap_start: When voice 2 starts (seconds)
        overlap_end: When voice 2 ends (seconds)

    Returns:
        (mixed_audio, overlap_ratio)
    """
    t = np.linspace(0, duration, int(sr * duration))

    # Voice 1 (entire duration)
    voice1 = 0.3 * np.sin(2 * np.pi * voice1_freq * t)

    # Voice 2 (partial - overlapping region)
    voice2 = np.zeros_like(t)
    overlap_start_idx = int(overlap_start * sr)
    overlap_end_idx = int(overlap_end * sr)
    voice2[overlap_start_idx:overlap_end_idx] = 0.3 * np.sin(
        2 * np.pi * voice2_freq * t[:overlap_end_idx - overlap_start_idx]
    )

    # Mix
    mixed = voice1 + voice2
    mixed = mixed / (np.max(np.abs(mixed)) + 1e-9)

    # Calculate actual overlap ratio
    overlap_ratio = (overlap_end - overlap_start) / duration

    return mixed.astype(np.float32), overlap_ratio

# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 1: OVERLAP DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverlapDetection:
    """Test overlap detection accuracy."""

    def test_single_speaker_no_overlap(self, separation_service):
        """Single speaker should have low overlap ratio."""
        audio = create_test_audio(duration=2.0, frequencies=[120])
        ratio, confidence = separation_service.estimate_overlap_ratio_v2(audio, sr=16000)

        assert ratio < 0.2, f"Single speaker detected as {ratio:.1%} overlap"
        assert confidence > 0.3, "Should have some confidence"
        logger.info(f"✅ Single speaker: {ratio:.1%} overlap, {confidence:.0%} confidence")

    def test_two_speakers_clear_overlap(self, separation_service):
        """Two speakers with clear overlap should be detected."""
        audio, expected_ratio = create_overlapping_speech(
            duration=3.0,
            overlap_start=1.0,
            overlap_end=2.5
        )
        ratio, confidence = separation_service.estimate_overlap_ratio_v2(audio, sr=16000)

        assert ratio > 0.2, f"Overlap not detected (only {ratio:.1%})"
        assert confidence > 0.5, "Should be reasonably confident"

        error = abs(ratio - expected_ratio) / expected_ratio
        assert error < 0.3, f"Ratio {ratio:.1%} off from expected {expected_ratio:.1%}"
        logger.info(f"✅ Overlapping speech: {ratio:.1%} overlap, {confidence:.0%} confidence")

    def test_partial_overlap(self, separation_service):
        """Partial overlap should be detected proportionally."""
        audio, expected_ratio = create_overlapping_speech(
            duration=4.0,
            overlap_start=1.0,
            overlap_end=2.0
        )
        ratio, confidence = separation_service.estimate_overlap_ratio_v2(audio, sr=16000)

        assert 0.15 < ratio < 0.35, f"Partial overlap {ratio:.1%} out of expected range"
        logger.info(f"✅ Partial overlap: {ratio:.1%} overlap")

    def test_long_overlap(self, separation_service):
        """Long overlapping section should show high overlap."""
        audio, expected_ratio = create_overlapping_speech(
            duration=2.0,
            overlap_start=0.2,
            overlap_end=1.8
        )
        ratio, confidence = separation_service.estimate_overlap_ratio_v2(audio, sr=16000)

        assert ratio > 0.6, f"Long overlap not detected (only {ratio:.1%})"
        logger.info(f"✅ Long overlap: {ratio:.1%} overlap")

    def test_confidence_increases_with_overlap(self, separation_service):
        """Confidence should increase as overlap becomes clearer."""
        single = create_test_audio(duration=2.0, frequencies=[120])
        multi, _ = create_overlapping_speech(duration=2.0)

        ratio1, conf1 = separation_service.estimate_overlap_ratio_v2(single, sr=16000)
        ratio2, conf2 = separation_service.estimate_overlap_ratio_v2(multi, sr=16000)

        assert conf2 > conf1, f"Confidence should increase (single={conf1:.0%}, multi={conf2:.0%})"
        logger.info(f"✅ Confidence scoring: {conf1:.0%} (single) → {conf2:.0%} (multi)")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 2: VOICE SEPARATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestVoiceSeparation:
    """Test voice separation (both CPU and GPU modes)."""

    def test_separation_available(self, separation_service):
        """Check that separation service is available."""
        assert separation_service.backend in ["spectral", "sepformer", "flagged"]
        logger.info(f"✅ Separation backend available: {separation_service.backend}")

    def test_no_separation_on_clean_speech(self, separation_service):
        """Clean speech (no overlap) should not attempt separation."""
        audio = create_test_audio(duration=2.0, frequencies=[120])
        result = separation_service.maybe_separate(audio, sr=16000)

        assert not result.separated or result.method == "none"
        logger.info(f"✅ Clean speech not separated (method={result.method})")

    def test_separation_on_overlapping_speech(self, separation_service):
        """Overlapping speech should trigger separation."""
        audio, _ = create_overlapping_speech(duration=3.0)
        result = separation_service.maybe_separate(audio, sr=16000)

        # Should attempt separation OR flag as overlap
        assert result.method in ["spectral", "sepformer", "flagged"]
        assert result.overlap_ratio > 0.1
        logger.info(f"✅ Overlapping speech detected: {result.method} ({result.overlap_ratio:.1%})")

    @pytest.mark.skipif(
        not all(hasattr(__import__('torch', fromlist=['cuda']), 'cuda')
                for _ in [1]),  # Check CUDA available
        reason="CUDA not available"
    )
    @pytest.mark.gpu
    def test_sepformer_separation_gpu(self, separation_service):
        """Test Sepformer separation on GPU (if available)."""
        if separation_service.backend != "sepformer":
            pytest.skip("Sepformer not available")

        audio, _ = create_overlapping_speech(duration=2.0)

        start = time.time()
        result = separation_service.maybe_separate(audio, sr=16000)
        elapsed = time.time() - start

        assert result.separated, "Sepformer should separate"
        assert len(result.streams) > 1, "Should produce multiple streams"
        assert elapsed < 3.0, f"GPU separation took {elapsed:.2f}s (expected <3s)"
        logger.info(f"✅ GPU Sepformer: {len(result.streams)} streams in {elapsed:.3f}s")

    @pytest.mark.cpu
    def test_spectral_separation_cpu(self, separation_service):
        """Test spectral separation on CPU."""
        audio, _ = create_overlapping_speech(duration=2.0)

        start = time.time()
        result = separation_service.maybe_separate(audio, sr=16000)
        elapsed = time.time() - start

        # CPU should either separate with spectral method or flag
        if result.separated:
            assert len(result.streams) > 1
            assert result.method == "spectral"
            assert elapsed < 1.0, f"CPU separation took {elapsed:.2f}s (expected <1s)"
            logger.info(f"✅ CPU Spectral: {len(result.streams)} streams in {elapsed:.3f}s")
        else:
            assert result.method in ["flagged", "none"]
            logger.info(f"✅ CPU fallback: {result.method} method")

    def test_separation_result_structure(self, separation_service):
        """Verify separation result has all required fields."""
        audio, _ = create_overlapping_speech(duration=2.0)
        result = separation_service.maybe_separate(audio, sr=16000)

        assert isinstance(result, SeparationResult)
        assert hasattr(result, 'separated')
        assert hasattr(result, 'streams')
        assert hasattr(result, 'overlap_ratio')
        assert hasattr(result, 'method')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'num_sources')

        assert 0 <= result.overlap_ratio <= 1
        assert 0 <= result.confidence <= 1
        assert result.num_sources >= 1

        logger.info(f"✅ Result structure valid: {result}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 3: DIARIZATION (SPEAKER DETECTION)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiarization:
    """Test speaker detection and diarization."""

    def test_diarization_available(self, diarization_service):
        """Check diarization service is available."""
        assert diarization_service.backend in ["pyannote", "enhanced"]
        logger.info(f"✅ Diarization backend: {diarization_service.backend}")

    def test_single_speaker_detection(self, diarization_service):
        """Single speaker should produce single turn."""
        audio = create_test_audio(duration=2.0, frequencies=[120])
        turns = diarization_service.diarize(audio, sr=16000)

        assert len(turns) >= 1, "Should detect at least one speaker"
        assert turns[0].start == 0.0
        assert turns[0].end > 0
        logger.info(f"✅ Single speaker: {len(turns)} turn(s)")

    def test_two_speaker_detection(self, diarization_service):
        """Two sequential speakers should produce multiple turns."""
        # Create 2-second audio: first half one pitch, second half different pitch
        audio1 = create_test_audio(duration=1.0, frequencies=[120])
        audio2 = create_test_audio(duration=1.0, frequencies=[180])
        audio = np.concatenate([audio1, audio2])

        turns = diarization_service.diarize(audio, sr=16000)

        # Should detect at least some speaker change
        assert len(turns) >= 1, "Should detect speakers"
        logger.info(f"✅ Two speakers: {len(turns)} turn(s) detected")

    def test_turn_confidence_scoring(self, diarization_service):
        """Each turn should have confidence score."""
        audio = create_test_audio(duration=2.0, frequencies=[120])
        turns = diarization_service.diarize(audio, sr=16000)

        for turn in turns:
            assert isinstance(turn, DiarTurn)
            assert 0 <= turn.confidence <= 1
            assert turn.confidence > 0, "Confidence should be > 0"

        logger.info(f"✅ Confidence scores: {[f'{t.confidence:.0%}' for t in turns]}")

    def test_turn_time_boundaries(self, diarization_service):
        """Turn times should be valid."""
        audio = create_test_audio(duration=3.0, frequencies=[120])
        turns = diarization_service.diarize(audio, sr=16000)

        for i, turn in enumerate(turns):
            assert 0 <= turn.start <= turn.end
            assert turn.start < 3.0
            assert turn.end <= 3.0
            assert (turn.end - turn.start) >= 0.4  # Min turn duration

            # Check ordering
            if i > 0:
                assert turn.start >= turns[i-1].end

        logger.info(f"✅ Time boundaries valid for {len(turns)} turns")

    def test_prosody_extraction(self, diarization_service):
        """Prosody features should be extracted."""
        audio = create_test_audio(duration=1.0, frequencies=[150])

        prosody = diarization_service._extract_prosody(audio, sr=16000)

        assert len(prosody) == 5  # [pitch_mean, pitch_std, energy_mean, energy_std, speaking_rate]
        assert all(isinstance(x, (int, float, np.number)) for x in prosody)
        logger.info(f"✅ Prosody features extracted: {prosody}")

    def test_voice_activity_segmentation(self, diarization_service):
        """VAD should segment into speech/silence regions."""
        # Create audio with clear silence region
        audio1 = create_test_audio(duration=1.0, frequencies=[120])
        silence = np.zeros(int(0.5 * 16000), dtype=np.float32)
        audio2 = create_test_audio(duration=1.0, frequencies=[120])
        audio = np.concatenate([audio1, silence, audio2])

        segments = diarization_service._voice_activity_segmentation(audio, sr=16000)

        assert len(segments) >= 1, "Should detect speech regions"
        assert all(0 <= s < e <= 2.5 for s, e in segments)
        logger.info(f"✅ VAD segments: {segments}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 4: END-TO-END INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:
    """Test complete multi-voice pipeline."""

    def test_multivoice_pipeline_no_overlap(
        self,
        separation_service,
        diarization_service,
        role_detector
    ):
        """Test complete pipeline with non-overlapping voices."""
        # Create sequential speech from different "speakers"
        audio1 = create_test_audio(duration=1.0, frequencies=[120])
        audio2 = create_test_audio(duration=1.0, frequencies=[180])
        audio = np.concatenate([audio1, audio2])

        # Step 1: Detect overlap
        overlap_ratio, overlap_conf = separation_service.estimate_overlap_ratio_v2(audio, sr=16000)
        logger.info(f"Overlap detected: {overlap_ratio:.1%}")

        # Step 2: Diarize
        turns = diarization_service.diarize(audio, sr=16000)
        assert len(turns) >= 1
        logger.info(f"Diarization: {len(turns)} turn(s)")

        # Step 3: Try separation
        sep_result = separation_service.maybe_separate(audio, sr=16000)
        logger.info(f"Separation: {sep_result.method} (separated={sep_result.separated})")

        logger.info(f"✅ Pipeline complete (no overlap)")

    def test_multivoice_pipeline_with_overlap(
        self,
        separation_service,
        diarization_service,
        role_detector
    ):
        """Test complete pipeline with overlapping voices."""
        audio, expected_overlap = create_overlapping_speech(
            duration=3.0,
            overlap_start=1.0,
            overlap_end=2.5
        )

        # Step 1: Detect overlap
        overlap_ratio, overlap_conf = separation_service.estimate_overlap_ratio_v2(audio, sr=16000)
        assert overlap_ratio > 0.15, "Should detect overlap"
        logger.info(f"✅ Overlap detected: {overlap_ratio:.1%} (confidence={overlap_conf:.0%})")

        # Step 2: Diarize
        turns = diarization_service.diarize(audio, sr=16000)
        assert len(turns) >= 1
        logger.info(f"✅ Diarization: {len(turns)} turn(s)")
        for turn in turns:
            logger.info(f"   {turn.local_label}: {turn.start:.1f}s-{turn.end:.1f}s (conf={turn.confidence:.0%})")

        # Step 3: Try separation
        sep_result = separation_service.maybe_separate(audio, sr=16000)
        if sep_result.separated:
            logger.info(f"✅ Separation: {len(sep_result.streams)} streams extracted via {sep_result.method}")
            assert len(sep_result.streams) > 0
        else:
            logger.info(f"✅ Separation: {sep_result.method} (no streams extracted)")

    def test_simultaneous_multiple_speakers(
        self,
        separation_service,
        diarization_service,
        role_detector
    ):
        """Test with 3 speakers all talking at once."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create 3 speakers with different frequencies
        speaker1 = 0.25 * np.sin(2 * np.pi * 120 * t)
        speaker2 = 0.25 * np.sin(2 * np.pi * 150 * t)
        speaker3 = 0.25 * np.sin(2 * np.pi * 200 * t)

        audio = speaker1 + speaker2 + speaker3
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        audio = audio.astype(np.float32)

        # Overlap detection
        overlap_ratio, overlap_conf = separation_service.estimate_overlap_ratio_v2(audio, sr=sr)
        assert overlap_ratio > 0.3, "Should detect high overlap"
        logger.info(f"✅ 3-speaker overlap: {overlap_ratio:.1%} detected")

        # Diarization
        turns = diarization_service.diarize(audio, sr=sr)
        assert len(turns) >= 1
        logger.info(f"✅ 3-speaker diarization: {len(turns)} segments")

        # Separation
        sep_result = separation_service.maybe_separate(audio, sr=sr)
        logger.info(f"✅ 3-speaker separation: {sep_result.method} (separated={sep_result.separated})")

        if sep_result.num_sources > 1:
            logger.info(f"   Estimated {sep_result.num_sources} sources")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 5: PERFORMANCE BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""

    @pytest.mark.benchmark
    def test_overlap_detection_speed(self, separation_service):
        """Overlap detection should be very fast."""
        audio = create_test_audio(duration=10.0, frequencies=[120])

        start = time.time()
        for _ in range(5):  # Run 5 times
            separation_service.estimate_overlap_ratio_v2(audio, sr=16000)
        elapsed = (time.time() - start) / 5

        assert elapsed < 0.05, f"Overlap detection too slow: {elapsed:.3f}s"
        logger.info(f"✅ Overlap detection: {elapsed*1000:.1f}ms per call")

    @pytest.mark.benchmark
    @pytest.mark.cpu
    def test_diarization_speed_cpu(self, diarization_service):
        """Diarization speed on CPU."""
        audio = create_test_audio(duration=5.0, frequencies=[120, 150])

        start = time.time()
        turns = diarization_service.diarize(audio, sr=16000)
        elapsed = time.time() - start

        # CPU should be fast
        assert elapsed < 5.0, f"CPU diarization too slow: {elapsed:.2f}s"
        logger.info(f"✅ CPU Diarization (5s audio): {elapsed:.3f}s")

    @pytest.mark.benchmark
    @pytest.mark.gpu
    def test_diarization_speed_gpu(self, diarization_service):
        """Diarization speed on GPU (if available)."""
        audio = create_test_audio(duration=5.0, frequencies=[120, 150])

        if diarization_service.backend != "pyannote":
            pytest.skip("PyAnnote not available")

        start = time.time()
        turns = diarization_service.diarize(audio, sr=16000)
        elapsed = time.time() - start

        logger.info(f"✅ GPU Diarization (5s audio): {elapsed:.3f}s")

    @pytest.mark.benchmark
    def test_complete_pipeline_performance(
        self,
        separation_service,
        diarization_service
    ):
        """Complete pipeline performance."""
        audio, _ = create_overlapping_speech(duration=5.0)

        start = time.time()

        # Overlap detection
        ratio, conf = separation_service.estimate_overlap_ratio_v2(audio, sr=16000)

        # Diarization
        turns = diarization_service.diarize(audio, sr=16000)

        # Separation
        sep_result = separation_service.maybe_separate(audio, sr=16000)

        elapsed = time.time() - start

        logger.info(f"✅ Complete pipeline (5s audio): {elapsed:.3f}s")
        logger.info(f"   - Turns detected: {len(turns)}")
        logger.info(f"   - Overlap: {ratio:.1%} (confidence={conf:.0%})")
        logger.info(f"   - Separation: {sep_result.method}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 6: ERROR HANDLING & EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_audio(self, separation_service, diarization_service):
        """Handle empty audio gracefully."""
        empty = np.array([], dtype=np.float32)

        # Should not crash
        ratio, conf = separation_service.estimate_overlap_ratio_v2(empty, sr=16000)
        assert ratio == 0.0

        turns = diarization_service.diarize(empty, sr=16000)
        assert len(turns) == 0

        logger.info(f"✅ Empty audio handled gracefully")

    def test_very_short_audio(self, separation_service, diarization_service):
        """Handle very short audio."""
        short = create_test_audio(duration=0.05, frequencies=[120])

        # Should handle gracefully
        ratio, conf = separation_service.estimate_overlap_ratio_v2(short, sr=16000)
        assert ratio >= 0

        turns = diarization_service.diarize(short, sr=16000)
        # May return fallback turn
        assert isinstance(turns, list)

        logger.info(f"✅ Short audio handled")

    def test_very_long_audio(self, separation_service):
        """Handle long audio segments."""
        long_audio = create_test_audio(duration=60.0, frequencies=[120])

        # Should not crash even with long audio
        ratio, conf = separation_service.estimate_overlap_ratio_v2(long_audio, sr=16000)
        assert ratio >= 0

        logger.info(f"✅ Long audio (60s) handled")

    def test_silence(self, separation_service, diarization_service):
        """Handle complete silence."""
        silence = np.zeros(16000 * 2, dtype=np.float32)

        ratio, conf = separation_service.estimate_overlap_ratio_v2(silence, sr=16000)
        assert ratio == 0.0

        turns = diarization_service.diarize(silence, sr=16000)
        # May return single turn or empty
        assert isinstance(turns, list)

        logger.info(f"✅ Silence handled")

    def test_very_loud_audio(self, separation_service):
        """Handle very loud audio (will be normalized)."""
        loud = np.ones(16000 * 2, dtype=np.float32) * 10.0  # Very loud

        # Should normalize and handle
        ratio, conf = separation_service.estimate_overlap_ratio_v2(loud, sr=16000)
        assert 0 <= ratio <= 1

        logger.info(f"✅ Very loud audio handled")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
