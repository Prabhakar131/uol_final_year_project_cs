from __future__ import annotations

import gc
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import pipeline
from cloudir.ai_models.model_worker import isolated_model_call


load_dotenv()


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pipeline_device() -> str | int:
    device = _device()

    if device == "cuda":
        return 0

    if device == "mps":
        return "mps"

    return -1


DEFAULT_VOICE_MODEL_ID = "openai/whisper-base"


def voice_model_id() -> str:
    return os.getenv("HF_VOICE_MODEL_ID", DEFAULT_VOICE_MODEL_ID)


def build_voice_pipeline(model_id: str):
    return pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=_pipeline_device(),
        torch_dtype=torch.float16 if _device() in {"mps", "cuda"} else torch.float32,
    )


# Whisper spells unfamiliar AWS names by sound ("GodDuty", "IAM principle",
# "Axesky"). Giving it the names learners use as a prompt fixes most of them.
AWS_VOCABULARY = "CloudTrail, GuardDuty, CloudWatch, IAM principal, access key, AssumeRole, StopLogging, MFA, source IP."


def whisper_generate_kwargs(voice_pipeline, *, vocabulary: str | None = AWS_VOCABULARY) -> dict[str, Any]:
    """
    Multilingual Whisper guesses each clip's language from its first seconds and
    has written accented English answers in Malay, or looped on one Malay word.
    Learners answer in English, so the language is fixed. English-only (.en)
    models take no language setting.
    """

    kwargs: dict[str, Any] = {}

    if getattr(voice_pipeline.model.generation_config, "is_multilingual", False):
        kwargs.update(language="english", task="transcribe")

    if vocabulary:
        prompt_ids = voice_pipeline.tokenizer.get_prompt_ids(vocabulary, return_tensors="pt")
        kwargs["prompt_ids"] = prompt_ids.to(voice_pipeline.model.device)

    return kwargs


# On a silent recording Whisper writes a filler phrase instead of nothing.
SILENCE_TRANSCRIPTS = {"", "you", "thank you", "thanks for watching"}


def transcript_text(result: Any) -> str:
    transcript = str(result.get("text", "") if isinstance(result, dict) else result).strip()

    if transcript.lower().strip(" .!") in SILENCE_TRANSCRIPTS:
        # The page adds ". Type your answer instead." after this message.
        raise ValueError("No speech was heard in the recording")

    return transcript


@lru_cache(maxsize=1)
def _load_voice_pipeline():
    return build_voice_pipeline(voice_model_id())


def clear_torch_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def unload_voice_model() -> None:
    _load_voice_pipeline.cache_clear()
    clear_torch_memory()


@isolated_model_call
def transcribe_learner_justification(audio_path: str | Path) -> dict[str, Any]:
    """
    Transcribes a learner's spoken justification into text.

    This stage supports the report's planned multimodal orchestration:
    action/evidence selection remains explicit, while speech captures the
    learner's reasoning for later security evaluation and coaching.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    voice_pipeline = _load_voice_pipeline()
    result = voice_pipeline(
        str(audio_path),
        return_timestamps=True,
        generate_kwargs=whisper_generate_kwargs(voice_pipeline),
    )

    transcript = transcript_text(result)

    return {
        "transcript": transcript,
        "model": voice_model_id(),
        "source": "spoken_justification",
        "audioFile": audio_path.name,
    }
