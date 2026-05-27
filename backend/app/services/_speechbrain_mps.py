"""
Workaround for the speechbrain 1.x bug that prevents loading models on MPS.

In `speechbrain/inference/interfaces.py::__init__`, `device_type` is only set
when the device is "cpu" or contains "cuda" — passing `device="mps"` raises
`AttributeError: 'X' object has no attribute 'device_type'` later during
inference. We bypass it by loading the model on CPU and then forcibly moving
every sub-module + the bookkeeping attributes to MPS.

Tested on Apple M2 with speechbrain 1.x:
  • SpeakerRecognition (ECAPA-TDNN): 2x faster, embedding identical (cos=1.0000)
  • SepformerSeparation:             3-5x faster, max-abs output diff < 4e-5
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def force_speechbrain_to_mps(model):
    """Patch a speechbrain Pretrained model so it runs end-to-end on MPS.

    Caller must have loaded the model on CPU first (`run_opts={"device": "cpu"}`).
    Returns the same model object for chaining.

    Quietly returns the model unchanged if torch/MPS isn't available — the
    caller can still use it on CPU.
    """
    try:
        import torch
        if not torch.backends.mps.is_available():
            return model
    except Exception:
        return model

    try:
        model.device = "mps"
        model.device_type = "mps"
        for name, module in list(model.mods.items()):
            if module is not None:
                model.mods[name] = module.to("mps")
        # MPS doesn't support autocast — keep the autocast context on cpu so
        # the `with model.inference_ctx:` blocks inside speechbrain remain no-ops.
        try:
            from speechbrain.utils.precision import TorchAutocast, AMPConfig
            precision_dtype = AMPConfig.from_name(model.precision).dtype
            model.inference_ctx = TorchAutocast(device_type="cpu", dtype=precision_dtype)
        except Exception:
            pass
        logger.info(f"✅ speechbrain model forced onto MPS via workaround patch")
    except Exception as e:
        logger.warning(f"force_speechbrain_to_mps failed ({e}); model stays on CPU")
    return model
