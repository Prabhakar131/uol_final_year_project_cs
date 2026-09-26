"""Model-like calls for tests/test_model_worker.py; the worker imports them by name."""
import os

from cloudir.ai_models.model_worker import isolated_model_call


class ExtractionLikeError(ValueError):
    pass


@isolated_model_call
def process_id(tag: str, *, extra: dict) -> dict:
    return {"pid": os.getpid(), "tag": tag, "extra": extra}


@isolated_model_call
def fail(message: str) -> None:
    raise ExtractionLikeError(message)


@isolated_model_call
def crash(code: int) -> None:
    os._exit(code)
