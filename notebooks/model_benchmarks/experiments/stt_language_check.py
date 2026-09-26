"""Checks the Whisper settings the app uses: English fixed, plus a list of AWS names.

The app's multilingual whisper-base guesses each clip's language. In the August STT
benchmark it wrote three of the eight recorded answers in Malay (one was a single
word repeated), while the English-only candidates did not. This runs Whisper the way the
app does (cloudir.ai_models.voice_model: MPS, float16) on:

  recorded   the eight learner recordings in stt_audio_samples (a real voice)
  synthetic  fresh answers spoken by seven macOS voices with different English
             accents; the words are known, so the word error rate is exact
  long       two synthetic answers over 30 seconds (the app records up to 45 s),
             which Whisper transcribes in 30-second windows

in three modes: "detect" (the old app behaviour, language guessed per clip),
"english" (language fixed) and "app" (language fixed plus the AWS_VOCABULARY
prompt, what voice_model now sends). English-only models have no language
setting, so they run without and with the prompt. Models are read from the local
cache only; nothing is downloaded.

The recordings do not follow the scripted sentences in STT_SAMPLE_DEFINITIONS
(speakers paraphrased), so their word error rate against that script is not
meaningful. For them this reports language, repeated-word loops, the benchmark's
own keyword recall and meaning-preserved rule, and agreement with the English-only
transcripts saved by the August benchmark.

    .venv/bin/python notebooks/model_benchmarks/experiments/stt_language_check.py [--models openai/whisper-base,openai/whisper-base.en]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS = ROOT / "notebooks/model_benchmarks"
RECORDINGS = BENCHMARKS / "stt_audio_samples/audio_samples/converted_wav"
AUGUST_RESULTS = BENCHMARKS / "results/stt_model_results.json"
RESULTS = BENCHMARKS / "results/experiments" / f"stt_language_check_{date.today():%Y%m%d}.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCHMARKS / "runners"))

from cloudir.ai_models.voice_model import AWS_VOCABULARY, build_voice_pipeline, clear_torch_memory, whisper_generate_kwargs  # noqa: E402
from run_benchmarks import STT_SAMPLE_DEFINITIONS, normalise_audio_stem, stt_keyword_matches, word_error_rate  # noqa: E402

VOICES = ["Daniel", "Rishi", "Aman", "Karen", "Moira", "Samantha", "Tessa"]
ANSWERS = [
    "The CloudTrail event shows the IAM principal calling StopLogging without MFA, so it directly supports containing that user.",
    "The GuardDuty finding is related, but it does not name the access key, so it only partly supports my action.",
    "The billing summary shows a cost spike, but it does not identify who made the change, so it is weak evidence.",
    "The CloudWatch Logs query links the source IP to the failed console logins, which supports blocking that address.",
    "The access key details show the key is still active and was last used from an unknown region, so I would deactivate it.",
    "The IAM activity summary lists the principal's recent API calls, but it does not show the event time for the AssumeRole call.",
    "I cannot prove data was accessed from this screenshot, so I would compare it with the CloudTrail record before escalating.",
    "The role session was created from an unfamiliar IP address shortly after the policy change, which suggests the credentials were stolen.",
]
TERMS = ["cloudtrail", "guardduty", "cloudwatch", "stoplogging", "assumerole", "principal", "access key", "iam", "mfa", "api"]
JOINED = [(r"cloud\s*trail", "cloudtrail"), (r"guard\s*duty", "guardduty"), (r"cloud\s*watch", "cloudwatch"),
          (r"stop\s*logging", "stoplogging"), (r"assume\s*role", "assumerole"), (r"\bi\s*a\s*m\b", "iam")]
DICTIONARY = {w.strip().lower() for w in Path("/usr/share/dict/words").read_text().split()} | {t for t in TERMS if " " not in t}


def normalise(text: str) -> str:
    text = text.lower().replace("-", " ").replace("'s", "s")
    for pattern, joined in JOINED:
        text = re.sub(pattern, joined, text)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())


def english_share(text: str) -> float:
    words = normalise(text).split()
    return round(sum(w in DICTIONARY for w in words) / len(words), 3) if words else 0.0


def longest_repeat(text: str) -> int:
    words, best, run = normalise(text).split(), 0, 0
    for i, word in enumerate(words):
        run = run + 1 if i and word == words[i - 1] else 1
        best = max(best, run)
    return best


def synthesise(folder: Path) -> list[dict]:
    clips = []
    for voice in VOICES:
        for number, text in enumerate(ANSWERS, 1):
            path = folder / f"{voice.lower()}_{number}.wav"
            subprocess.run(["say", "-v", voice, "-o", str(path), "--data-format=LEI16@16000", text], check=True)
            clips.append({"set": "synthetic", "id": path.stem, "voice": voice, "file": path, "reference": text})
    for voice in VOICES[:2]:
        path, text = folder / f"{voice.lower()}_long.wav", " ".join(ANSWERS[:5])
        subprocess.run(["say", "-v", voice, "-o", str(path), "--data-format=LEI16@16000", text], check=True)
        clips.append({"set": "long", "id": path.stem, "voice": voice, "file": path, "reference": text})
    return clips


def recordings() -> list[dict]:
    august = json.loads(AUGUST_RESULTS.read_text()) if AUGUST_RESULTS.exists() else []
    english_only = {row["sample_id"]: row["transcript"] for row in august if row["model_id"] == "openai/whisper-base.en"}
    clips = []
    for path in sorted(RECORDINGS.glob("*.wav")):
        definition = STT_SAMPLE_DEFINITIONS.get(normalise_audio_stem(path.stem), {})
        sample_id = definition.get("id", path.stem)
        clips.append({"set": "recorded", "id": sample_id, "file": path, "keywords": definition.get("keywords", []),
                      "august_english_only": english_only.get(sample_id, "")})
    return clips


def transcribe(recogniser, path: Path, generate_kwargs: dict) -> tuple[str, str, float]:
    started = time.perf_counter()
    result = recogniser(str(path), return_timestamps=True, return_language=True, generate_kwargs=dict(generate_kwargs))
    elapsed = time.perf_counter() - started
    languages = sorted({str(chunk.get("language")) for chunk in result.get("chunks", []) if chunk.get("language")})
    return str(result.get("text", "")).strip(), ",".join(languages), round(elapsed, 3)


def score(clip: dict, transcript: str, language: str, seconds: float) -> dict:
    row = {"set": clip["set"], "id": clip["id"], "transcript": transcript, "language": language, "seconds": seconds,
           "english_share": english_share(transcript), "longest_repeat": longest_repeat(transcript)}
    row["wrong_language"] = row["english_share"] < 0.6 or (bool(language) and language != "english")
    row["loop"] = row["longest_repeat"] >= 4
    said = normalise(transcript)
    if clip["set"] != "recorded":
        reference = normalise(clip["reference"])
        row |= {"voice": clip["voice"], "reference": clip["reference"], "wer": word_error_rate(reference, said),
                "terms_missed": [t for t in TERMS if re.search(rf"\b{t}\b", reference) and not re.search(rf"\b{t}\b", said)]}
    else:
        matched = [k for k in clip["keywords"] if stt_keyword_matches(k, transcript.lower())]
        recall = len(matched) / len(clip["keywords"]) if clip["keywords"] else 0.0
        row |= {"keyword_recall": round(recall, 3), "meaning_preserved": recall >= 0.5,
                "wer_vs_august_english_only": word_error_rate(normalise(clip["august_english_only"]), said)
                if clip["august_english_only"] else None}
    return row


def summarise(rows: list[dict]) -> dict:
    summary = {}
    for name in ("recorded", "synthetic", "long"):
        part = [r for r in rows if r["set"] == name]
        if not part:
            continue
        mean = lambda key: round(sum(r[key] for r in part) / len(part), 3)  # noqa: E731
        entry = {"clips": len(part), "wrong_language": sum(r["wrong_language"] for r in part),
                 "loops": sum(r["loop"] for r in part), "mean_seconds": mean("seconds")}
        if name == "recorded":
            entry |= {"meaning_preserved": sum(r["meaning_preserved"] for r in part), "mean_keyword_recall": mean("keyword_recall"),
                      "mean_wer_vs_august_english_only": mean("wer_vs_august_english_only")}
        else:
            missed = [t for r in part for t in r["terms_missed"]]
            entry |= {"mean_wer": mean("wer"), "exact": sum(r["wer"] == 0 for r in part),
                      "terms_missed": {t: missed.count(t) for t in sorted(set(missed))},
                      "mean_wer_by_voice": {v: round(sum(r["wer"] for r in mine) / len(mine), 3)
                                            for v in VOICES if (mine := [r for r in part if r["voice"] == v])}}
        summary[name] = entry
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="openai/whisper-base")
    args = parser.parse_args()
    runs = []
    with tempfile.TemporaryDirectory(prefix="stt_language_check_") as folder:
        clips = recordings() + synthesise(Path(folder))
        for model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
            try:
                recogniser = build_voice_pipeline(model_id)
            except OSError as exc:
                print(f"{model_id}: not in the local cache, skipped ({exc.__class__.__name__})", flush=True)
                runs.append({"model": model_id, "skipped": "not in the local Hugging Face cache"})
                continue
            english, app = whisper_generate_kwargs(recogniser, vocabulary=None), whisper_generate_kwargs(recogniser)
            modes = [("detect", {}), ("english", english), ("app", app)] if english else [("no prompt", {}), ("app", app)]
            for mode, generate_kwargs in modes:
                recogniser(str(clips[0]["file"]), return_timestamps=True, generate_kwargs=dict(generate_kwargs))  # warm-up
                rows = [score(clip, *transcribe(recogniser, clip["file"], generate_kwargs)) for clip in clips]
                settings = {k: v for k, v in generate_kwargs.items() if k != "prompt_ids"} | (
                    {"prompt": AWS_VOCABULARY} if "prompt_ids" in generate_kwargs else {})
                run = {"model": model_id, "mode": mode, "settings": settings, "summary": summarise(rows), "rows": rows}
                runs.append(run)
                print(model_id, mode, json.dumps(run["summary"], indent=1), flush=True)
            del recogniser
            clear_torch_memory()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"date": f"{date.today():%Y-%m-%d}", "voices": VOICES, "answers": ANSWERS, "runs": runs}, indent=2))
    print(f"Saved {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
