"""Runs PyTorch model calls in short-lived worker processes.

PyTorch's Apple GPU (MPS) backend caches a compiled graph for every tensor shape
it runs and never frees them, and text generation meets new shapes at every
token. Unloading a model does not release that cache: one scenario build left
8.6 GB of CPU memory in the process, and two builds pushed the server to 20 GB
on a 24 GB Mac. A worker process that exits returns all of it.

The cache also grows within one call: a next-turn-sized generation (3.5k-token
prompt, 800-token answer) takes its worker from 3.6 to 12.3 GB. So each model
call gets its own worker; wrapping the whole next-turn loop (generation, review
and retries) in one worker reached 16 GB. A static KV cache did not help: it was
slower (10.0 vs 11.4 tokens/s) and used more memory.

The server turns this on (CLOUDIR_ISOLATE_MODELS=1). Tests and scripts run
models in-process unless they set it, so mocks and patches keep working.
The security model runs in llama.cpp, which does not keep this cache.
"""
from __future__ import annotations

import functools
import importlib
import os
import pickle
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, TypeVar

ISOLATE_ENV = "CLOUDIR_ISOLATE_MODELS"
CODE_ROOT = Path(__file__).resolve().parents[2]
F = TypeVar("F", bound=Callable[..., Any])


def isolated_model_call(func: F) -> F:
    """Runs func in a fresh worker process when isolation is on, otherwise in-process."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if os.environ.get(ISOLATE_ENV) != "1":
            return func(*args, **kwargs)
        return _run_in_worker(func.__module__, func.__qualname__, args, kwargs)

    return wrapper  # type: ignore[return-value]


def _run_in_worker(module: str, name: str, args: tuple, kwargs: dict) -> Any:
    with tempfile.TemporaryDirectory(prefix="cloudir-model-") as directory:
        request, result = Path(directory) / "request.pkl", Path(directory) / "result.pkl"
        request.write_bytes(pickle.dumps((module, name, args, kwargs)))
        # The worker runs the call directly; its output joins the server log.
        process = subprocess.run([sys.executable, "-m", __name__, str(request), str(result)],
                                 cwd=CODE_ROOT, env={**os.environ, ISOLATE_ENV: "0"})

        if not result.exists():
            raise RuntimeError(f"The model worker for {name} exited with code {process.returncode} "
                               "before returning a result.")

        try:
            succeeded, value = pickle.loads(result.read_bytes())
        except Exception as exc:
            raise RuntimeError(f"The model worker for {name} returned an unreadable result: {exc}") from exc

    if succeeded:
        return value
    raise value


def _serve(request_path: str, result_path: str) -> None:
    module, name, args, kwargs = pickle.loads(Path(request_path).read_bytes())
    func: Any = importlib.import_module(module)
    for part in name.split("."):
        func = getattr(func, part)

    try:
        outcome = (True, func(*args, **kwargs))
    except Exception as exc:
        traceback.print_exc()  # the full traceback stays in the server log
        outcome = (False, exc)

    try:
        payload = pickle.dumps(outcome)
    except Exception:
        payload = pickle.dumps((False, RuntimeError(f"{name} failed in the model worker: {outcome[1]!r}")))
    Path(result_path).write_bytes(payload)


if __name__ == "__main__":
    _serve(sys.argv[1], sys.argv[2])
