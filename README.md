# CloudIR Trainer

Adaptive AI Cloud Incident Response Trainer developed for the University of London final year project.

## Project Snapshot

CloudIR Trainer is a local Flask web application that turns ACSE-Eval cloud-security cases into interactive five-turn incident response exercises.

The learner selects an incident-response action and supporting evidence, then records or types a justification. A vision-language model reads the evidence screenshot, a security model evaluates the decision, and a coach model provides feedback and generates the next turn. The application tracks visibility, containment, risk, phase, and action history before producing a final debrief.

### Implemented features

- Browser interface built with HTML, CSS, and JavaScript.
- Streamed ACSE-Eval dataset preparation.
- Three enabled scenarios: `identity-management`, `automated-security-response`, and `cost-management`.
- Five-turn adaptive incident lifecycle from initial triage through recovery.
- Generated AWS-style IAM, CloudTrail, CloudWatch, GuardDuty, access-key, billing, and IAM-activity evidence.
- Action and evidence selection with typed or recorded justification.
- Local speech-to-text transcription.
- Streamed VLM, security-reasoning, and coach-model evaluation.
- AI-generated turns with validation, evidence repair, progression safeguards, quality scoring, and up to three generation attempts.
- Per-turn evaluation traces, progress indicators, and final debrief.
- Dataset status, clean artifact rebuild, scenario reset, and health endpoints.
- Versioned saved runs, automatic checkpoints, Test Mode save/leave/resume, independent retry attempts, and ZIP export.
- Model-comparison notebook and saved VLM, security, coach, and STT benchmark results.

## Current Application Models

These are the current application models. Benchmark files compare additional candidates.

| Role | Hugging Face model | Purpose |
|---|---|---|
| Vision | `Qwen/Qwen2.5-VL-3B-Instruct` | Read evidence screenshots and architecture diagrams |
| Security | `fdtn-ai/Foundation-Sec-8B-Instruct`, run as the official 8-bit GGUF `fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF` | Evaluate the action, evidence, and learner reasoning |
| Coach | `Qwen/Qwen2.5-1.5B-Instruct` | Feedback, turn generation, quality checks, and final debrief |
| Speech-to-text | `openai/whisper-base`, language fixed to English, with a prompt of AWS names | Transcribe spoken justifications |

The vision, coach and speech loaders choose Apple MPS first, then CUDA, then CPU, using float16 on MPS/CUDA and float32 on CPU. The security model runs through llama.cpp (Metal on Apple silicon, CUDA or CPU elsewhere) from its 8-bit GGUF file. Generation is greedy (`do_sample=False`, or temperature 0 in llama.cpp), and models are unloaded between pipeline stages to reduce memory use.

Whisper is told the answer is English. Left to guess, the multilingual `whisper-base` wrote three of the eight benchmark recordings in Malay; with English fixed and a short prompt of AWS names (`AWS_VOCABULARY` in `cloudir/ai_models/voice_model.py`) it wrote all eight in English, and 50 of 56 synthetic answers in seven English accents word for word ([results](notebooks/model_benchmarks/results/experiments/stt_language_check_20260927.json)). A silent recording returns "No speech was heard" rather than Whisper's filler word. Transcribing adds the transcript after anything already typed, and both stay editable before submission.

## Configuration

Copy `.env.example` to `.env` beside `app.py`. It holds these settings:

```dotenv
HF_IMAGE_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
HF_SECURITY_MODEL_ID=fdtn-ai/Foundation-Sec-8B-Instruct
HF_COACH_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
HF_VOICE_MODEL_ID=openai/whisper-base
HF_NEXT_TURN_MAX_NEW_TOKENS=1600

SECURITY_MODEL_BACKEND=llama_cpp
HF_SECURITY_GGUF_REPO=fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF
HF_SECURITY_GGUF_FILE=foundation-sec-8b-instruct-q8_0.gguf
HF_SECURITY_GGUF_CONTEXT=8192
```

`.env` is ignored by Git. Do not commit tokens or credentials. The image, security, and coach IDs are required. The voice model otherwise defaults to `openai/whisper-base`.

By default the security model runs from an 8-bit GGUF copy through llama.cpp
(about 8.5 GB, downloaded on first use) instead of the full 16-bit weights (about
16 GB), which swap heavily on a 24 GB Mac. `HF_SECURITY_MODEL_ID` is still
required: both backends format prompts with its tokenizer.

| Environment variable | Default or effective value |
|---|---|
| `SECURITY_MODEL_BACKEND` | `llama_cpp` (quantized GGUF); set `transformers` to load the full 16-bit weights |
| `HF_SECURITY_GGUF_REPO` / `HF_SECURITY_GGUF_FILE` | `fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF` / `foundation-sec-8b-instruct-q8_0.gguf` |
| `HF_SECURITY_GGUF_PATH` | Unset; a local `.gguf` path that skips the download |
| `HF_SECURITY_GGUF_CONTEXT` | 8192 tokens |
| `HF_MAX_NEW_TOKENS` | 350 for coach feedback; security evaluation enforces a minimum of 600 |
| `HF_SECURITY_NORMALISER_MAX_NEW_TOKENS` | 700 |
| `HF_IMAGE_MAX_NEW_TOKENS` | 650 for general evidence; CloudWatch extraction enforces a minimum of 1400; 900 default for architecture parsing |
| `HF_FINAL_DEBRIEF_MAX_NEW_TOKENS` | 1800 |
| `HF_NEXT_TURN_MAX_NEW_TOKENS` | Minimum 3200 for next-turn generation; code raises the configured 1600 to this minimum |
| `HF_INITIAL_TURN_MAX_NEW_TOKENS` | Falls back to the configured next-turn value, currently 1600 |
| `HF_QUALITY_JUDGE_MAX_NEW_TOKENS` | 650 |
| `ACSE_ARCH_VLM_MAX_SIDE` | 1280 pixels |
| `COACH_MODEL_BACKEND` | `llama_cpp`: the coach runs from a 16-bit GGUF with transformers' full-context repetition penalty (1.1, from the model's generation config), 3–8× faster than PyTorch at comparable output; set `transformers` for PyTorch |
| `HF_COACH_GGUF_REPO` / `HF_COACH_GGUF_FILE` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` / `qwen2.5-1.5b-instruct-fp16.gguf` |
| `HF_COACH_GGUF_PATH` | Unset; a local `.gguf` path that skips the download |
| `HF_COACH_GGUF_CONTEXT` | 16384 tokens |
| `HF_IMAGE_DTYPE` | `bfloat16` on the GPU (float16 overflowed on some screenshots); set `float16` for GPUs without bfloat16 |
| `HF_IMAGE_MPS_RELEASE_EVERY` | 32: the VLM returns freed Apple GPU memory every 32 generated tokens |
| `CLOUDIR_ISOLATE_MODELS` | `1` when the server runs (`python app.py`): the VLM, coach and speech models run in a worker process that exits after each call, so PyTorch's Apple GPU graph cache cannot build up in the server (server memory stayed at 0.34–0.42 GB across the automated sessions); `0` runs them in-process (the default for tests and scripts) |

## Application Flow

1. The frontend checks which ACSE-Eval scenarios and generated assets are ready.
2. Preparation selects a case, parses its architecture, normalises its threat model, creates runtime JSON, writes the incident timeline, generates turn 1, and renders evidence screenshots.
3. The learner chooses an action and evidence and provides a typed or spoken justification.
4. The VLM analyses the screenshot, the security model assesses the decision, and the coach returns feedback.
5. State and evaluation traces are saved. The coach generates and validates the next turn while progression safeguards keep the incident coherent.
6. After turn 5, the application creates the final debrief and shows it as a printable incident response report.

The interface in `frontend/` follows reduced-motion preferences (and has its own **Reduce motion** switch),
works with a keyboard and screen reader, and includes a **Guide** page that walks through each screen.

Before judgement, extraction receives structural checks. CloudWatch requires
structured, readable event messages (or a readable zero-result query); the
matched-record badge need not equal displayed rows. Failed extraction is retried
once with a corrective instruction. If it still fails, evaluation stops without
a verdict or progress award and the run can be retried. Query metadata is passed
separately from event observations. CloudWatch verdicts cite event row numbers;
invalid citations or a Strong verdict without a readable cited row are rejected.
These checks do not prove that every field was transcribed correctly or that
every model explanation is sound.

Learner reasoning is assessed using references to complete, numbered transcript
statements. The application—not the model—renders the exact quoted statement
with a constrained assessment (`supported`, `overclaim`, or `needs_detail`).
Instructor questions are separate and cannot become learner statements. An
empty transcript gets no reasoning credit, even if the selected evidence is
strong. Per-turn coach feedback explains the evidence without paraphrasing or
praising learner reasoning. Invalid statement references and detected personal
attributions trigger one retry; persistent failures stop evaluation without
progress. Statement ratings still require model judgement, and the prose check
is conservative rather than a general proof of semantic correctness.

CloudWatch fact normalisation preserves supplied events and normal/OK signals;
it does not add incident events to fill a table. Missing fields remain Unknown.
Screenshots wrap complete messages and grow to show every supplied row, with
matched-record and displayed-row counts shown separately. Coordinate-only PNG
metadata guides the VLM's enlarged pixel crop; it contains no expected answers.
Malformed ISO timestamps trigger the extraction retry, but plausible incorrect
timestamps cannot be detected by this structural check.

### Incident timeline evidence

During preparation the security model writes one incident timeline for the
scenario (`data/runtime/incident_timeline.json`): the attacker, background and
responder principals, their AWS API events, a GuardDuty finding and daily
cost. Every turn's evidence is built from it by code
(`cloudir/scenario/incident_timeline.py`):

- **strong**: the attacker's incident steps up to that turn (turn 5 shows the responder's recovery)
- **partial**: the same principal's normal work the day before, from its usual IP (right principal, suspicious step missing)
- **weak**: other principals' routine read-only activity, an unrelated low-severity finding, or flat spend

The coach still writes each turn's briefing, actions, titles and "why it may
matter" text around this evidence. Evidence summaries are captions generated
from the rendered facts, so they cannot claim anything the screenshot lacks.

The timeline is validated before use. Event names must be real CloudTrail
eventNames: full lists are checked for IAM, STS, sign-in, CloudTrail, S3, EC2,
GuardDuty, Security Hub, EventBridge, Step Functions and Lambda, and a pattern
is checked elsewhere. The prompt lists real eventNames for the scenario's own
services, and a rejected name is answered with its closest real ones (for
example `UpdateRolePolicy` → `PutRolePolicy`). Principals must be ARNs,
findings must use real GuardDuty types, and the attacker's steps must use the
services named by the threat model. Every turn's strong, partial and weak
evidence must also pass the support-role content check. A rejected timeline is
shown back to the model with the reasons so it fixes only those, up to three
attempts. Code, not the model, does the bookkeeping: it assigns turns when the
model's labels are incomplete, and it generates the principal's baseline. It
also keeps only routine background calls that do not touch the suspect,
topping them up when too few remain. It renames names that give the answer
away, such as `malicioususer` or `malicious-lambda`, everywhere they appear,
and adds the version CloudTrail records on Lambda eventNames. Screenshots no longer print
conclusions: CloudTrail shows "Resources referenced" instead of an
"Investigation signal", and IAM shows "Last activity" instead of "Risk flags".
Scenarios prepared before this change keep their generated evidence.

The security judge also receives the current turn's briefing and known
context, exactly as the learner sees them. This lets it tell the principal under
investigation apart from someone else's routine activity. The prompt marks this
as context, not evidence.

Evidence support roles are no longer reassigned from template or list position.
Invalid role sets and duplicate templates are rejected instead of silently
relabelling or replacing content. Next-turn generation uses its existing retry;
invalid initial-turn preparation can fail and require another attempt. This
checks structure, not whether every authored partial/weak label is defensible.
Historical saved runs are unchanged.

## Repository Structure

```text
uol_final_year_project_cs/
├── app.py                     Flask application and API routes
├── frontend/                  Browser UI
├── cloudir/
│   ├── ai_models/             Vision, security, coach, voice, and lifecycle code
│   ├── dataset_preparation/   ACSE-Eval preparation pipeline
│   ├── evidence/              Screenshot generation and AWS-style templates
│   ├── evidence_core/         Schemas, template inference, and fact repair
│   ├── scenario/              State, turn generation, validation, and progression
│   ├── services/              Dataset, evaluation, cleanup, and debrief services
│   └── paths.py               Shared filesystem paths
├── data/
│   ├── acse_eval.jsonl        ACSE-Eval case records
│   ├── source/                Architecture images and threat models
│   ├── processed/             Prepared scenario inputs
│   ├── prepared/              Built Turn 1 of each scenario, restored by Start Scenario
│   ├── runtime/               Active working copy of turns and evaluations
│   └── runs/                  Saved runs and immutable checkpoints (local, ignored by Git)
├── output/generated_evidence/ Rendered evidence screenshots
├── notebooks/                 Model evaluation notebook and benchmarks
├── tests/                     Storage and API regression tests (no model inference)
├── setup.sh / setup.bat       One-time setup (environment, packages, .env, models)
├── start.sh / start.bat       Starts the app and opens it in the browser
├── download_models.py         Downloads every model before the first run
├── requirements.txt           Python dependencies
├── .env.example               Model configuration to copy to .env
└── .gitignore                 Local environment and cache exclusions
```

## Setup on a New Computer

### Requirements

- **Python 3.10, 3.11 or 3.12, 64-bit** (3.11 is tested). The pinned PyTorch has no builds for Python 3.13 or newer. On Windows, install it from [python.org](https://www.python.org/downloads/).
- **About 25 GB of free disk space** for the packages and models, and ideally 24 GB of memory (see the measurements below).
- **Windows: nothing else.** `setup.bat` installs only ready-built packages, so no compiler is needed, and it installs the Microsoft Visual C++ Redistributable if it is missing. Keep the project in a short folder such as `C:\cloudir` (see [Windows troubleshooting](#windows-troubleshooting)).
- **macOS and Linux: a C/C++ compiler**, because `llama-cpp-python` compiles during installation: the Xcode Command Line Tools on macOS (`xcode-select --install`) or `build-essential` on Ubuntu/Debian. Recording a spoken answer also needs `ffmpeg` (`brew install ffmpeg` or `sudo apt install ffmpeg`); typed answers work without it.

### Quick start

Clone the repository (or use **Code → Download ZIP** on GitHub and unzip it):

```bash
git clone https://github.com/Prabhakar131/uol_final_year_project_cs.git
cd uol_final_year_project_cs
```

On macOS or Linux:

```bash
bash setup.sh
bash start.sh
```

On Windows, double-click `setup.bat` once, then `start.bat`. Or run them from the project folder in Command Prompt:

```bat
setup.bat
start.bat
```

In PowerShell, put `.\` in front:

```powershell
.\setup.bat
.\start.bat
```

The setup script runs once. It finds a supported Python, creates the `.venv` environment, installs the packages, creates `.env` from `.env.example`, and downloads the models (about 20 GB). `--skip-models` (`bash setup.sh --skip-models`, `setup.bat --skip-models` or `.\setup.bat --skip-models`) does everything except the model download. Nothing is missing afterwards, but each model then downloads the first time the app needs it, so the first evaluation takes much longer.

On Windows, `setup.bat` installs `llama-cpp-python` 0.3.19 from the project's ready-built Windows wheels instead of compiling the 0.3.35 in `requirements.txt`, which macOS and Linux build. A Turn 1 evaluation gave the same verdict and word-for-word the same coach feedback on both versions. It also puts a copy of `ffmpeg` in `.venv\Scripts` for recorded answers.

The start script runs the app and opens `http://127.0.0.1:5000` in the browser once it is ready. Press Ctrl+C to stop it. Use `127.0.0.1` rather than `localhost`: on macOS, AirPlay Receiver answers `localhost:5000`. If another program uses port 5000, choose another port with `CLOUDIR_PORT=5050 bash start.sh` (on Windows, `set CLOUDIR_PORT=5050`, then `start.bat`).

All three scenarios come already built (`data/prepared/`), so **Start Scenario** works without running the preparation step.

### Manual setup

The scripts run these steps, which can also be run by hand. On macOS or Linux:

```bash
python3.11 -m venv .venv
cp .env.example .env
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_models.py
.venv/bin/python app.py
```

On Windows, in PowerShell from the project folder. These commands call the environment's own Python, so nothing needs activating:

```powershell
py -3.11 -m venv .venv
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
Get-Content requirements.txt | Where-Object { $_ -notmatch '^llama-cpp-python' } | Set-Content "$env:TEMP\cloudir-requirements.txt"
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r "$env:TEMP\cloudir-requirements.txt"
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python==0.3.19 imageio-ffmpeg==0.6.0
.\.venv\Scripts\python.exe -c "import imageio_ffmpeg, shutil; shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), r'.venv\Scripts\ffmpeg.exe')"
.\.venv\Scripts\python.exe download_models.py
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
.\.venv\Scripts\python.exe app.py
```

`download_models.py` is optional in both, as with `--skip-models`.

### Windows troubleshooting

- **"This folder's path is … characters long", or a "No such file or directory" error while installing.** Windows limits file paths to 260 characters, and the installed packages use about 160 of them inside the project folder. Move the folder somewhere short, such as `C:\cloudir`, delete `.venv`, and run `setup.bat` again. Extracting the GitHub ZIP with File Explorer nests the folder twice under Downloads, which can be too long. Alternatively, an administrator can enable long paths by running `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force` in an administrator PowerShell, then restarting Windows.
- **"No matching distribution found".** Python must be 64-bit 3.10, 3.11 or 3.12. Delete `.venv`, install Python 3.11 (64-bit), and run `setup.bat` again.
- **"DLL load failed".** Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). `setup.bat` does this itself when the runtime files are missing.
- **Transcribe says "ffmpeg was not found".** Start the app with `start.bat`, which puts the bundled `ffmpeg` on the path. If you set up before this was added, run `setup.bat` again.
- **Slow evaluations.** Without an NVIDIA GPU, every model runs on the CPU, so an evaluation takes much longer than on Apple silicon or a GPU. It still completes.

### Models

The model weights are not stored in this repository: together they are about 20 GB, far beyond GitHub's file limits. They are public on Hugging Face, so no account or access token is needed. `download_models.py` fetches all of them into the normal Hugging Face cache (`~/.cache/huggingface/hub`), showing progress, and running it again only fetches what is missing:

| Model | Download |
|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` (vision) | 7.5 GB |
| `fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF` (security) | 8.5 GB |
| `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, fp16 file (coach) | 3.6 GB |
| `openai/whisper-base` (speech-to-text) | 0.3 GB |

The script is optional: without it, each model downloads the first time a scenario needs it, and the first evaluation waits on that download without showing progress. Allow about 25 GB of free disk space. `.env`, `.venv`, and the model cache stay on the local computer.

On macOS and Linux, `llama-cpp-python` builds with Metal on Apple silicon and for the CPU elsewhere. To build it for an NVIDIA GPU, set `CMAKE_ARGS="-DGGML_CUDA=on"` before running `setup.sh`. On Windows, `setup.bat` installs the ready-built CPU version.

Measured on a 24 GB Apple-silicon Mac (24 Sep 2026): the 8-bit security model uses about 9–10 GB. It normalised the identity-management threat model in 18 s and replayed a saved evaluation in 13 s, without growing swap. The 16-bit weights stalled the same normalisation for more than 3 minutes while swapping. The vision model's CloudWatch extraction is now the memory peak, at about 15 GB of GPU memory, so machines with less than 24 GB may still swap during that step.

## Main API Routes

| Route | Purpose |
|---|---|
| `GET /api/dataset/status` | Report source and generated-file readiness |
| `POST /api/dataset/prepare-stream` | Prepare a scenario with streamed progress |
| `POST /api/dataset/flush-generated` | Remove generated artifacts for a clean rebuild |
| `GET /api/state` | Load the current turn and its evidence |
| `POST /api/transcribe-justification` | Transcribe learner audio |
| `POST /api/evaluate-stream` | Stream VLM, security, coach, and next-turn stages |
| `POST /api/evaluate` | Run evaluation as one response |
| `POST /api/continue` | Move to the generated next turn |
| `GET /api/final-debrief` | Return the scenario debrief |
| `GET /api/incident-timeline` | Return the incident timeline for the final report (only once the scenario is complete) |
| `POST /api/reset` | Archive the current run and start a separate attempt from the same Turn 1 |
| `GET /api/health` | Confirm the backend is running |

## Test Mode and Saved Runs

Enable **Test Mode** using the top-right button, then start a prepared scenario.
The run toolbar is available throughout the scenario:

| Control | Behaviour |
|---|---|
| Run name + **Save Run** | Save a named checkpoint, including selected choices and the typed/transcribed justification. |
| **Save & Leave** | Save the unfinished run and return to scenario selection. |
| **Save & Leave When Ready** | During evaluation, queue leaving until the AI evaluation and next-turn preparation finish. It does not cancel inference. Keep the tab open until saving finishes. |
| **Retry This Turn** | Save the current attempt and create a separate run from its before-evaluation checkpoint (or initial starting point). The same starting state and evidence are restored. |
| **Saved Runs** | Review evaluations, resume a run, repeat a selected starting checkpoint, or export a ZIP. |

For the evidence-choice experiment: choose one action and evidence, submit, then use
**Retry This Turn** and select different evidence while keeping the action and
justification consistent. Results from both attempts remain available in Saved Runs.
Later AI-generated turns may differ; a retry preserves the chosen starting point,
not a promise of identical future model outputs.

Runs are stored under `data/runs/run-<id>/`. Each checkpoint includes runtime JSON,
the generated evidence images, available scenario source/processed data, progress,
draft text, configured model IDs/token limits, and the Git revision/local-change
flag. Evaluations record state before/after each turn. Operation events record
elapsed times and streamed stage outputs; next-turn generation records its retries,
quality reviews and fallback outcome. A checkpoint has checksums for its saved files.
Model weight revisions are not pinned, and a Git revision with local changes is not
a full source-code archive.

Progress is checkpointed before evaluation and after successful evaluation/continue.
The active checkpoint is restored when Flask restarts. Failed/interrupted evaluations
retain diagnostic snapshots and recover the previous checkpoint. Unsaved browser
edits and untranscribed audio recordings are not recoverable after closing a tab;
transcribe recordings and use **Save Run** or **Save & Leave** first.

Only one operation can use the local scenario workspace at a time. Saving, resetting
or switching runs while inference is active is rejected by the server. Run this as
one local Flask process; the operation lock is not a multi-worker deployment lock.

Every successful build also keeps a copy of that scenario's Turn 1 starting point
in `data/prepared/<scenario>/`. Building or resuming another scenario replaces the
shared workspace, but **Start Scenario** restores the selected scenario's copy, so
switching scenarios needs no rebuild. A failed rebuild keeps the last working copy.
A workspace built before copies existed is copied before anything replaces it.

**Clean Restart** removes disposable working artifacts, not `data/runs/` or
`backups/`. It also deletes the cleaned scenario's prepared copy (all copies for a
full clean), so that scenario must be rebuilt. Cleaning another scenario preserves
the active run. The scenario picker reports a scenario as prepared when it has a
copy or the workspace holds it. Old working data
with no reliable saved state is archived as **Existing working data (unverified)**;
it can be reviewed/exported but is not presented as a resumable completed run.

Saved runs and backups are excluded from Git. Use **Export ZIP** to keep or submit
a selected run; there is currently no ZIP-import interface. Snapshots copy their
files rather than deduplicating them, so disk usage grows with saved checkpoints.

Run the storage/API regression tests without model inference:

```bash
.venv/bin/python -m unittest discover -s tests -v
node --test tests/test_run_controls.cjs
```

Tests use a temporary workspace (`CLOUDIR_WORKSPACE_DIR`) and stub AI responses;
they verify run management, not model accuracy or full AI-session reliability.

## Model Evaluation Artifacts

`notebooks/model_comparison_and_evaluation.ipynb` presents the model evaluation. Runners under `notebooks/model_benchmarks/runners/` perform VLM, security, coach, and STT comparisons. Current outputs live in `notebooks/model_benchmarks/results/`, while older results are retained under `results/superseded/`. See `notebooks/README.md` for commands and artifact locations.

## Snapshot Notes

- Runtime JSON and generated evidence are working artifacts and may change as scenarios are prepared or played.
- Flask uses one active working session; its saved checkpoint restores progress after a server restart.
- Model IDs are recorded, but exact Hugging Face weight revisions are not pinned.
- The Flask development server uses `debug=True` and is intended for local use.
- GitHub contains committed code and artifacts, but not `.env`, `.venv`, model caches, or uncommitted local files.
