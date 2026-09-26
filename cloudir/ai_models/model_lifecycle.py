from __future__ import annotations

import gc
import os
import warnings

import torch

from cloudir.ai_models.coach_model import unload_coach_model
from cloudir.ai_models.image_model import unload_image_model
from cloudir.ai_models.security_model import unload_security_model
from cloudir.ai_models.voice_model import unload_voice_model


def unload_all_models() -> None:
    """Unloads the local AI models in a fixed order."""

    unload_voice_model()
    unload_image_model()
    unload_security_model()
    unload_coach_model()


def log_memory(tag: str) -> None:
    """
    Prints how much memory the local AI models are using, so leaks between
    stages are visible in the server output.

    Process RSS is misleading on Apple silicon (GPU memory is unified), so this
    reports MPS allocations, the process's real memory footprint and which model
    objects are still alive.
    """

    parts: list[str] = []

    try:
        if torch.backends.mps.is_available():
            parts.append(
                f"mps_alloc={torch.mps.current_allocated_memory() / 1e9:.1f}GB "
                f"mps_driver={torch.mps.driver_allocated_memory() / 1e9:.1f}GB"
            )
    except Exception:
        pass

    try:
        import psutil

        info = psutil.Process(os.getpid()).memory_full_info()
        footprint = getattr(info, "phys_footprint", None) or getattr(info, "uss", 0)
        parts.append(f"footprint={footprint / 1e9:.1f}GB")
    except Exception:
        pass

    alive: list[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for obj in gc.get_objects():
                if isinstance(obj, torch.nn.Module) and "For" in type(obj).__name__:
                    size = sum(p.numel() for p in obj.parameters()) / 1e9
                    alive.append(f"{type(obj).__name__}({size:.1f}B)")
    except Exception:
        pass

    # llama.cpp models are not torch modules, so the scan above cannot see them.
    from cloudir.ai_models import security_model

    if security_model._gguf_model is not None:
        alive.append("SecurityGGUF(llama.cpp)")

    parts.append("models_alive=" + (", ".join(alive) if alive else "none"))

    print(f"[memory] {tag}: " + " ".join(parts), flush=True)
