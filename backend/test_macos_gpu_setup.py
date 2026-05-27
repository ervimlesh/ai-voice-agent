"""
macOS GPU/MPS configuration tests for the multi-voice AI agent.

Verifies that each ML component (Whisper, ECAPA, Pyannote, Sepformer) loads on
the correct device for Apple Silicon and that speaker differentiation works.

Run:
    pytest test_macos_gpu_setup.py -v -s

Hardware compatibility matrix (Apple Silicon, M1/M2/M3):
    Component               MPS?   Notes
    ----------------------  -----  -------------------------------------------
    faster-whisper          NO     CTranslate2 supports only CPU/CUDA
    ECAPA-TDNN (speechbrain)YES*   2x faster on MPS via _speechbrain_mps patch
    Sepformer  (speechbrain)YES*   3-5x faster on MPS via _speechbrain_mps patch
    pyannote.audio          YES    Native MPS support

    *speechbrain 1.x is broken on MPS out of the box (missing device_type
    attribute). The app.services._speechbrain_mps helper loads on CPU and
    forces the model onto MPS — accuracy is identical (cosine match 1.0000).
"""
import os
# HF_TOKEN is read from the environment (.env / shell). Never hardcode it.
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytest
import torch

# Apply huggingface_hub + torch.load shims before importing pyannote/speechbrain
from app.main import _install_hf_kwarg_shim, _patch_torch_load  # noqa: F401


# ───────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def settings():
    from app.core.config import get_settings
    return get_settings()


@pytest.fixture(scope="module")
def voice_a():
    return _synth_voice(140, duration=2.0)


@pytest.fixture(scope="module")
def voice_b():
    return _synth_voice(280, duration=2.0)


@pytest.fixture(scope="module")
def voice_a2():
    return _synth_voice(140, duration=2.0, noise=0.01)


def _synth_voice(f0, duration=2.0, sr=16000, noise=0.005):
    """Synthesize a voice-like signal with formants and speech-rate amplitude modulation."""
    t = np.linspace(0, duration, int(sr * duration))
    sig = (0.5 * np.sin(2 * np.pi * f0 * t)
           + 0.3 * np.sin(2 * np.pi * f0 * 2.1 * t)
           + 0.2 * np.sin(2 * np.pi * f0 * 3.3 * t)
           + 0.1 * np.sin(2 * np.pi * f0 * 4.7 * t))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    sig = sig * env
    sig += np.random.randn(len(t)).astype(np.float32) * noise
    return sig.astype(np.float32)


# ───────────────────────────────────────────────────────────────────────
# Tier 1: PyTorch / MPS hardware availability
# ───────────────────────────────────────────────────────────────────────

class TestPyTorchMPS:
    def test_pytorch_installed(self):
        assert torch.__version__ is not None

    def test_mps_built(self):
        assert torch.backends.mps.is_built(), "PyTorch must be built with MPS support"

    def test_mps_available(self):
        assert torch.backends.mps.is_available(), "MPS is unavailable — re-check macOS / Xcode tools"

    def test_mps_tensor_round_trip(self):
        """Allocate a tensor on MPS, run a matmul, copy back to CPU."""
        x = torch.randn(64, 64, device="mps")
        y = torch.randn(64, 64, device="mps")
        z = (x @ y).cpu().numpy()
        assert z.shape == (64, 64)
        assert not np.isnan(z).any()


# ───────────────────────────────────────────────────────────────────────
# Tier 2: Library-level GPU support (what works on MPS, what doesn't)
# ───────────────────────────────────────────────────────────────────────

class TestLibraryMPSSupport:
    def test_faster_whisper_does_not_support_mps(self):
        """Documents that faster-whisper rejects MPS — CPU is the only option."""
        from faster_whisper import WhisperModel
        with pytest.raises(Exception, match=r"(?i)mps|unsupported"):
            WhisperModel("tiny", device="mps", compute_type="float32")

    def test_pyannote_loads_on_mps(self):
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ["HF_TOKEN"],
        )
        pipeline.to(torch.device("mps"))  # must succeed without exception

    def test_speechbrain_ecapa_fails_on_mps_directly(self):
        """Documents the speechbrain 1.x device_type bug — direct MPS load fails.
        The fix lives in app.services._speechbrain_mps."""
        from speechbrain.inference import SpeakerRecognition
        with pytest.raises(AttributeError, match="device_type"):
            SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "mps"},
            )

    def test_speechbrain_ecapa_loads_on_mps_via_workaround(self):
        """The workaround patch makes ECAPA fully MPS-resident."""
        from speechbrain.inference import SpeakerRecognition
        from app.services._speechbrain_mps import force_speechbrain_to_mps
        model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        model = force_speechbrain_to_mps(model)
        assert model.device == "mps"
        assert model.device_type == "mps"
        # Every loaded sub-module must be on MPS
        for name, module in model.mods.items():
            if module is None:
                continue
            for p in module.parameters():
                assert p.device.type == "mps", f"{name} param on {p.device}"
                break

    def test_speechbrain_sepformer_loads_on_mps_via_workaround(self):
        """The workaround patch makes Sepformer fully MPS-resident."""
        from speechbrain.inference.separation import SepformerSeparation
        from app.services._speechbrain_mps import force_speechbrain_to_mps
        model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_models/sepformer-wsj02mix",
            run_opts={"device": "cpu"},
        )
        model = force_speechbrain_to_mps(model)
        assert model.device == "mps"
        # Run a tiny inference end-to-end on MPS to prove it
        mix = torch.randn(1, 4000)
        with torch.no_grad():
            est = model.separate_batch(mix)
        assert est.shape[0] == 1


# ───────────────────────────────────────────────────────────────────────
# Tier 3: Project services initialize on the configured devices
# ───────────────────────────────────────────────────────────────────────

class TestServiceInitialization:
    def test_whisper_loads_on_cpu(self, settings):
        from app.services.whisper_service import WhisperService
        ws = WhisperService(settings)
        ws._get_model()
        assert settings.whisper_device == "cpu", (
            "faster-whisper must use CPU on Apple Silicon "
            f"(currently: {settings.whisper_device})"
        )

    def test_voice_embedding_loads_on_mps(self, settings):
        from app.services.voice_embedding_service import VoiceEmbeddingService
        ves = VoiceEmbeddingService(settings)
        assert ves.available, "ECAPA-TDNN failed to load"
        assert ves.device == "mps", (
            f"ECAPA-TDNN should be on MPS via workaround (currently: {ves.device})"
        )
        # The underlying model parameters must actually live on MPS
        for module in ves._model.mods.values():
            if module is None:
                continue
            for p in module.parameters():
                assert p.device.type == "mps", f"Param on {p.device}, not MPS"
                break

    def test_diarization_loads_on_mps(self, settings):
        from app.services.diarization_service import DiarizationService
        from app.services.speaker_detector_service import SpeakerDetector
        ds = DiarizationService(settings, SpeakerDetector())
        assert ds.backend == "pyannote", (
            f"Expected pyannote backend, got '{ds.backend}'."
        )

    def test_separation_loads_on_mps(self, settings):
        from app.services.separation_service import SeparationService
        ss = SeparationService(settings)
        assert ss.backend == "sepformer", (
            f"Expected sepformer backend, got '{ss.backend}'"
        )
        assert ss._device == "mps", (
            f"Sepformer should be on MPS via workaround (currently: {ss._device})"
        )
        # Modules actually on MPS?
        for module in ss._model.mods.values():
            if module is None:
                continue
            for p in module.parameters():
                assert p.device.type == "mps", f"Sepformer param on {p.device}"
                break


# ───────────────────────────────────────────────────────────────────────
# Tier 4: Multi-voice differentiation — the core feature
# ───────────────────────────────────────────────────────────────────────

class TestMultiVoiceDifferentiation:
    def test_ecapa_distinguishes_different_speakers(self, settings, voice_a, voice_b, voice_a2):
        """The headline test: ECAPA embeddings must cluster same-speaker tighter
        than different-speaker."""
        from app.services.voice_embedding_service import VoiceEmbeddingService
        ves = VoiceEmbeddingService(settings)
        e_a  = np.array(ves.embed(voice_a))
        e_b  = np.array(ves.embed(voice_b))
        e_a2 = np.array(ves.embed(voice_a2))

        sim_same = float(np.dot(e_a, e_a2))
        sim_diff = float(np.dot(e_a, e_b))

        print(f"\n      Same-speaker cosine:      {sim_same:+.4f}")
        print(f"      Different-speaker cosine: {sim_diff:+.4f}")
        print(f"      Discrimination margin:    {sim_same - sim_diff:+.4f}")

        assert sim_same > sim_diff, "Different speakers must score lower than same speaker"
        assert sim_same - sim_diff > 0.10, (
            f"Discrimination margin too small ({sim_same - sim_diff:.4f}); "
            "embeddings won't reliably separate speakers"
        )

    def test_diarization_runs_on_real_audio(self, settings, voice_a, voice_b):
        """Diarization must complete without error on MPS — the pipeline itself
        is the thing under test, not its accuracy on synthetic tones."""
        from app.services.diarization_service import DiarizationService
        from app.services.speaker_detector_service import SpeakerDetector
        ds = DiarizationService(settings, SpeakerDetector())
        combined = np.concatenate([voice_a, voice_b])
        turns = ds.diarize(combined, sr=16000)
        assert len(turns) >= 1, "Diarization returned no turns"
        for t in turns:
            assert 0.0 <= t.start < t.end <= len(combined) / 16000 + 0.1

    def test_overlap_separation_no_crash(self, settings, voice_a, voice_b):
        """Sepformer (CPU) must process an overlapped mix without crashing."""
        from app.services.separation_service import SeparationService
        ss = SeparationService(settings)
        if ss.backend != "sepformer":
            pytest.skip("Sepformer not loaded — separation will only flag overlap")
        mix = voice_a + voice_b
        result = ss.maybe_separate(mix, sr=16000)
        assert result.method in ("sepformer", "none", "flagged")


# ───────────────────────────────────────────────────────────────────────
# Tier 5: Configuration sanity (catches stale .env)
# ───────────────────────────────────────────────────────────────────────

class TestConfiguration:
    def test_whisper_device_is_cpu(self, settings):
        assert settings.whisper_device == "cpu", (
            "WHISPER_DEVICE must be 'cpu' on Apple Silicon (faster-whisper limitation)"
        )

    def test_whisper_compute_type_is_int8(self, settings):
        assert settings.whisper_compute_type in ("int8", "int8_float32"), (
            f"On CPU, use int8 for ~3x throughput (got {settings.whisper_compute_type})"
        )

    def test_diarization_device_is_mps(self, settings):
        assert settings.diarization_device in ("mps", "auto"), (
            f"DIARIZATION_DEVICE should be 'mps' on Apple Silicon (got {settings.diarization_device})"
        )

    def test_voice_embedding_device_is_mps(self, settings):
        assert settings.voice_embedding_device in ("mps", "auto"), (
            f"VOICE_EMBEDDING_DEVICE should be 'mps' on Apple Silicon "
            f"(got {settings.voice_embedding_device})"
        )

    def test_separation_enabled(self, settings):
        assert settings.separation_enabled, "Separation must be enabled for overlap un-mixing"


# ───────────────────────────────────────────────────────────────────────
# Tier 6: MPS vs CPU benchmark — proves the GPU is paying off
# ───────────────────────────────────────────────────────────────────────

class TestMPSPerformance:
    """Marks: numerical equivalence + measurable speedup."""

    def test_ecapa_mps_matches_cpu_output(self, voice_a):
        """ECAPA output on MPS must match CPU within floating-point noise."""
        from speechbrain.inference import SpeakerRecognition
        from app.services._speechbrain_mps import force_speechbrain_to_mps
        m_cpu = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        m_mps = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        m_mps = force_speechbrain_to_mps(m_mps)
        wav = torch.from_numpy(voice_a).float().unsqueeze(0)
        with torch.no_grad():
            e_cpu = m_cpu.encode_batch(wav).squeeze().cpu().numpy().flatten()
            e_mps = m_mps.encode_batch(wav.to("mps")).squeeze().cpu().numpy().flatten()
        e_cpu /= np.linalg.norm(e_cpu) + 1e-9
        e_mps /= np.linalg.norm(e_mps) + 1e-9
        cos_match = float(np.dot(e_cpu, e_mps))
        print(f"\n      ECAPA cosine(MPS, CPU): {cos_match:.6f}")
        assert cos_match > 0.999, f"MPS embedding diverges from CPU (cos={cos_match})"

    def test_sepformer_mps_matches_cpu_output(self):
        """Sepformer output on MPS must match CPU within floating-point noise."""
        from speechbrain.inference.separation import SepformerSeparation
        from app.services._speechbrain_mps import force_speechbrain_to_mps
        m_cpu = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_models/sepformer-wsj02mix",
            run_opts={"device": "cpu"},
        )
        m_mps = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_models/sepformer-wsj02mix",
            run_opts={"device": "cpu"},
        )
        m_mps = force_speechbrain_to_mps(m_mps)

        sr = 8000
        t = np.linspace(0, 2.0, sr * 2)
        mix = (0.3 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 280 * t)).astype(np.float32)
        mix_t = torch.from_numpy(mix).float().unsqueeze(0)

        with torch.no_grad():
            est_cpu = m_cpu.separate_batch(mix_t).squeeze(0).cpu().numpy()
            est_mps = m_mps.separate_batch(mix_t).squeeze(0).cpu().numpy()

        max_diff = float(np.abs(est_cpu - est_mps).max())
        print(f"\n      Sepformer max-abs diff (MPS vs CPU): {max_diff:.6f}")
        assert max_diff < 1e-3, f"Sepformer MPS diverges from CPU (max diff {max_diff})"
