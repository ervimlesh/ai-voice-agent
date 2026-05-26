import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env into os.environ so HF Hub and other libs can find HF_TOKEN
load_dotenv()

# Compatibility shim for huggingface_hub: pyannote.audio 3.x uses the old
# `use_auth_token=` kwarg, but huggingface_hub renamed it to `token=`.
# Patch the function at multiple binding sites since pyannote captures
# direct references at module-import time.
def _install_hf_kwarg_shim() -> None:
    """Translate pyannote's use_auth_token= kwarg to huggingface_hub's token=.
    Patch the function on huggingface_hub itself AND on every site-package module
    that has already imported it by direct binding (e.g. `from huggingface_hub
    import hf_hub_download`). Without the per-module patch, pyannote keeps the
    pre-shim reference and the kwarg never gets rewritten."""
    try:
        import sys
        import huggingface_hub as _hf
        from huggingface_hub import file_download as _file_download

        _patched_marker = "_voiceagent_kwarg_shim"

        def _kwarg_adapter(fn):
            def _wrapped(*args, **kwargs):
                if "use_auth_token" in kwargs and "token" not in kwargs:
                    kwargs["token"] = kwargs.pop("use_auth_token")
                return fn(*args, **kwargs)
            try:
                _wrapped._voiceagent_kwarg_shim = True
            except Exception:
                pass
            return _wrapped

        def _already_patched(fn) -> bool:
            try:
                return getattr(fn, _patched_marker, False)
            except Exception:
                return False

        # Patch the root module and the inner file_download module
        for _mod in (_hf, _file_download):
            for _name in ("hf_hub_download", "snapshot_download", "cached_download"):
                if hasattr(_mod, _name):
                    _fn = getattr(_mod, _name)
                    if not _already_patched(_fn):
                        setattr(_mod, _name, _kwarg_adapter(_fn))

        # Patch already-imported modules that captured a direct binding.
        # Force-load pyannote.audio.core.pipeline so we can rebind its symbol.
        try:
            import pyannote.audio.core.pipeline as _pa_pipeline
            if hasattr(_pa_pipeline, "hf_hub_download"):
                _fn = _pa_pipeline.hf_hub_download
                if not _already_patched(_fn):
                    _pa_pipeline.hf_hub_download = _kwarg_adapter(_fn)
        except Exception:
            pass

        # Targeted sweep — only patch already-loaded pyannote/speechbrain modules
        # that captured a direct binding. Skipping the generic sys.modules walk
        # avoids triggering noisy transformers deprecation warnings.
        target_names = {"hf_hub_download", "snapshot_download"}
        target_prefixes = ("pyannote", "speechbrain", "lightning", "pytorch_lightning")
        for _mod_name, _mod in list(sys.modules.items()):
            if _mod is None or not _mod_name.startswith(target_prefixes):
                continue
            if _mod is _hf or _mod is _file_download:
                continue
            for _name in target_names:
                try:
                    _fn = getattr(_mod, _name, None)
                    if _fn is None or _already_patched(_fn):
                        continue
                    if _fn is getattr(_hf, _name, None) or _fn is getattr(_file_download, _name, None):
                        setattr(_mod, _name, _kwarg_adapter(_fn))
                except Exception:
                    continue
    except Exception as _e:
        print(f"huggingface_hub shim skipped: {_e}")

_install_hf_kwarg_shim()

# PyTorch 2.6+ changed torch.load() default to weights_only=True, which breaks
# pyannote's pickled checkpoints. Allowlist the safe globals it needs, and
# patch torch.load to default weights_only=False (pyannote models come from
# huggingface — a trusted source we authenticate against).
def _patch_torch_load() -> None:
    try:
        import torch
        try:
            from torch.torch_version import TorchVersion
            torch.serialization.add_safe_globals([TorchVersion])
        except Exception:
            pass
        try:
            from omegaconf.listconfig import ListConfig
            from omegaconf.dictconfig import DictConfig
            torch.serialization.add_safe_globals([ListConfig, DictConfig])
        except Exception:
            pass

        _orig_load = torch.load
        if not getattr(_orig_load, "_voiceagent_patched", False):
            def _wrapped_load(*args, **kwargs):
                # Force weights_only=False if it's None or not set.
                # PyTorch 2.6+ defaults to True; lightning passes None.
                if kwargs.get("weights_only") is None:
                    kwargs["weights_only"] = False
                return _orig_load(*args, **kwargs)
            _wrapped_load._voiceagent_patched = True
            torch.load = _wrapped_load
            # Patch lightning_fabric internal too (captured reference)
            try:
                import lightning_fabric.utilities.cloud_io as _cloud_io
                if hasattr(_cloud_io, "torch"):
                    _cloud_io.torch.load = _wrapped_load
            except Exception:
                pass
    except Exception as _e:
        print(f"torch.load patch skipped: {_e}")

_patch_torch_load()

# Compatibility shims for pyannote.audio: torchaudio 2.11 removed several legacy APIs.
# Provide minimal stubs so pyannote can import and run.
try:
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        @dataclass
        class AudioMetaData:
            sample_rate: int = 16000
            num_frames: int = 0
            num_channels: int = 1
            bits_per_sample: int = 16
            encoding: str = "PCM_S"
        torchaudio.AudioMetaData = AudioMetaData

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: None

    if not hasattr(torchaudio, "info"):
        def _info_shim(filepath, *args, **kwargs):
            import soundfile as _sf
            with _sf.SoundFile(str(filepath)) as f:
                meta = torchaudio.AudioMetaData(
                    sample_rate=f.samplerate,
                    num_frames=len(f),
                    num_channels=f.channels,
                    bits_per_sample=16,
                    encoding="PCM_S",
                )
                return meta
        torchaudio.info = _info_shim
except Exception as _e:
    print(f"torchaudio shim skipped: {_e}")

from app.api.v1.router import api_router
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Export HF_TOKEN to env vars HuggingFace libraries look for
if settings.hf_token:
    os.environ["HF_TOKEN"] = settings.hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = settings.hf_token


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
