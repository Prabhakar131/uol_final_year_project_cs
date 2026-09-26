"""Download every model the application uses before the first run.

Otherwise each model downloads the first time a scenario needs it (about 20 GB
in total), and the first evaluation waits on that without showing progress.
Run once after creating .env:

    python download_models.py

Files go to the normal Hugging Face cache, where the application finds them.
Running it again only fetches what is missing.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download

if not (Path(__file__).parent / ".env").exists():
    sys.exit("Create .env first: copy .env.example to .env.")
load_dotenv()

from cloudir.ai_models.coach_model import DEFAULT_COACH_GGUF_FILE, DEFAULT_COACH_GGUF_REPO, coach_backend
from cloudir.ai_models.security_model import DEFAULT_GGUF_FILE, DEFAULT_GGUF_REPO, security_backend
from cloudir.ai_models.voice_model import voice_model_id

# Tokenizer, chat template and configs: all a llama.cpp backend needs from the original repo.
CONFIG_FILES = ["*.json", "*.txt", "*.jinja"]
WEIGHTS = CONFIG_FILES + ["*.safetensors"]


def required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        sys.exit(f"{name} is missing from .env.")
    return value


def gguf(label: str, path_var: str, repo_var: str, default_repo: str, file_var: str, default_file: str) -> None:
    if os.getenv(path_var):
        print(f"{label}: using {os.environ[path_var]} ({path_var})")
        return
    repo, filename = os.getenv(repo_var, default_repo), os.getenv(file_var, default_file)
    print(f"{label}: {repo}/{filename}")
    hf_hub_download(repo, filename)


def main() -> None:
    image_id = required("HF_IMAGE_MODEL_ID")
    security_id = required("HF_SECURITY_MODEL_ID")
    coach_id = required("HF_COACH_MODEL_ID")

    print(f"Vision model: {image_id} (about 7.5 GB)")
    snapshot_download(image_id, allow_patterns=WEIGHTS)

    if security_backend() == "llama_cpp":
        print(f"Security model tokenizer: {security_id}")
        snapshot_download(security_id, allow_patterns=CONFIG_FILES)
        gguf("Security model (about 8.5 GB)", "HF_SECURITY_GGUF_PATH", "HF_SECURITY_GGUF_REPO", DEFAULT_GGUF_REPO,
             "HF_SECURITY_GGUF_FILE", DEFAULT_GGUF_FILE)
    else:
        print(f"Security model: {security_id} (about 16 GB)")
        snapshot_download(security_id, allow_patterns=WEIGHTS)

    if coach_backend() == "llama_cpp":
        print(f"Coach model tokenizer: {coach_id}")
        snapshot_download(coach_id, allow_patterns=CONFIG_FILES)
        gguf("Coach model (about 3.6 GB)", "HF_COACH_GGUF_PATH", "HF_COACH_GGUF_REPO", DEFAULT_COACH_GGUF_REPO,
             "HF_COACH_GGUF_FILE", DEFAULT_COACH_GGUF_FILE)
    else:
        print(f"Coach model: {coach_id} (about 3 GB)")
        snapshot_download(coach_id, allow_patterns=WEIGHTS)

    print(f"Speech-to-text model: {voice_model_id()} (about 0.3 GB)")
    snapshot_download(voice_model_id(), allow_patterns=WEIGHTS)

    print("All models are downloaded.")


if __name__ == "__main__":
    main()
