# Benchmark Runners

Maintained command-line entry points for model evaluation.

- `run_benchmarks.py` contains shared benchmark cases, model specs, loaders,
  scoring helpers, and the combined runner.
- `security_model_evaluation.py`, `stt_model_evaluation.py`,
  `coach_model_evaluation.py`, and `vlm_model_evaluation.py` are focused CLI
  wrappers for each model family.

Run these from the project root, for example:

```bash
python3 notebooks/model_benchmarks/runners/vlm_model_evaluation.py --metadata-only
```
