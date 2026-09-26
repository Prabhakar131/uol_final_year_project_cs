# CloudIR Trainer Notebooks

This folder contains the notebook evidence for the report deliverable.

- `model_comparison_and_evaluation.ipynb` is the cleaned model evaluation notebook. It can launch focused benchmark runners and then reads the generated artifacts.
- `model_benchmarks/runners/run_benchmarks.py` runs the Hugging Face model comparisons and writes CSV/JSON files to `model_benchmarks/results/`.
- `model_benchmarks/runners/security_model_evaluation.py` runs the security-reasoning model batches outside the notebook kernel.
- `model_benchmarks/runners/stt_model_evaluation.py` runs speech-to-text evaluations outside the notebook kernel.
- `model_benchmarks/runners/coach_model_evaluation.py` runs coach feedback evaluations outside the notebook kernel.
- `model_benchmarks/runners/vlm_model_evaluation.py` runs screenshot VLM evaluations outside the notebook kernel.

## Folder Layout

```text
notebooks/
  model_comparison_and_evaluation.ipynb
  model_benchmarks/
    *_model_evaluation.py       Maintained benchmark entry points.
    run_benchmarks.py           Shared benchmark library and combined runner.
    experiments/                One-off diagnostics and scenario experiments.
    results/                    Current CSV/JSON/Markdown benchmark outputs.
      experiments/              Outputs from one-off experiment scripts.
      superseded/               Older runs kept for provenance.
    stt_audio_samples/          Local STT sample audio inputs.
```

Keep the current result files directly under `model_benchmarks/results/` because
the evaluation notebook reads them from that location. Move old comparison runs
or diagnostic outputs into `results/superseded/` or `results/experiments/` once
they are no longer the active report artifacts.

The benchmark runner supports multiple candidates for each AI task: VLM screenshot analysis, speech-to-text, security reasoning, and coach feedback.

The notebook is designed to run from Jupyter without starting the Flask app. It reads the existing runtime artifacts under `data/runtime/`, generated screenshots under `output/generated_evidence/`, and model configuration from `.env`.

For a full Hugging Face comparison run:

```bash
python3 notebooks/model_benchmarks/runners/run_benchmarks.py --allow-downloads --include-vlm --include-coach --include-stt
```

## Security Model Evaluation Workflow

Run heavy security-model tests from the terminal, then read the generated CSV,
JSON, or Markdown files from the notebook.

For a metadata-only planning report:

```bash
python3 notebooks/model_benchmarks/runners/security_model_evaluation.py --metadata-only
```

For a quick smoke test of the current application security model:

```bash
python3 notebooks/model_benchmarks/runners/security_model_evaluation.py --limit-cases 2
```

For an explicit comparison between cached candidates:

```bash
python3 notebooks/model_benchmarks/runners/security_model_evaluation.py \
  --models foundation-sec-8b-instruct,qwen2.5-1.5b-instruct \
  --limit-cases 4
```

The focused runner writes:

```text
notebooks/model_benchmarks/results/security_model_focused_results.csv
notebooks/model_benchmarks/results/security_model_focused_summary.csv
notebooks/model_benchmarks/results/security_model_evaluation_report.md
```

## STT, Coach, And VLM Evaluation Workflow

Run these tasks separately. STT is usually quickest, coach is moderate, and VLM
should be run last because screenshot models are slow locally.

For speech-to-text:

```bash
python3 notebooks/model_benchmarks/runners/stt_model_evaluation.py \
  --models whisper-base,whisper-tiny-en,whisper-base-en,whisper-small \
  --limit-samples 2
```

For coach feedback:

```bash
python3 notebooks/model_benchmarks/runners/coach_model_evaluation.py \
  --models qwen2.5-1.5b-instruct,qwen2.5-0.5b-instruct,flan-t5-small,flan-t5-base \
  --limit-cases 2
```

For VLM screenshot reading:

```bash
python3 notebooks/model_benchmarks/runners/vlm_model_evaluation.py \
  --models qwen2.5-vl-3b-instruct,qwen2-vl-2b-instruct,llava-onevision-0.5b,florence-2-base-ft \
  --limit-cases 1 \
  --max-new-tokens 80
```

The task runners write:

```text
notebooks/model_benchmarks/results/stt_model_results.csv
notebooks/model_benchmarks/results/stt_model_summary.csv
notebooks/model_benchmarks/results/stt_model_evaluation_report.md
notebooks/model_benchmarks/results/coach_model_results.csv
notebooks/model_benchmarks/results/coach_model_summary.csv
notebooks/model_benchmarks/results/coach_model_evaluation_report.md
notebooks/model_benchmarks/results/vlm_model_results.csv
notebooks/model_benchmarks/results/vlm_model_summary.csv
notebooks/model_benchmarks/results/vlm_model_evaluation_report.md
```
