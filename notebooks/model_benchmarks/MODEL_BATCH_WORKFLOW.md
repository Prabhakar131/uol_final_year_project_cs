# CloudIR Model Batch Workflow

This file records the local model workflow so temporary benchmark downloads can
be deleted without losing the production configuration.

## Production Models

These are the models configured for the application runtime:

```text
HF_SECURITY_MODEL_ID=fdtn-ai/Foundation-Sec-8B-Instruct
HF_IMAGE_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
HF_COACH_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
HF_VOICE_MODEL_ID=openai/whisper-base
SECURITY_MODEL_BACKEND=llama_cpp
HF_SECURITY_GGUF_REPO=fdtn-ai/Foundation-Sec-8B-Instruct-Q8_0-GGUF
HF_SECURITY_GGUF_FILE=foundation-sec-8b-instruct-q8_0.gguf
```

They can be redownloaded later from Hugging Face if their local cache folders
are removed.

Since 24 Sep 2026 the application runs the security model from its official
8-bit GGUF file (about 8.5 GB). Only the tokenizer files of
`fdtn-ai/Foundation-Sec-8B-Instruct` stay cached; its 16-bit weights were
removed. The security benchmark loads candidate models at full precision
through transformers, so benchmarking Foundation-Sec-8B-Instruct again
redownloads about 16 GB. Earlier benchmark results were measured at 16-bit.
Experiment scripts that call the application's security model now run the
8-bit file and record it through `security_model_label()`.

## Recommended Batch Process

1. Edit `notebooks/model_comparison_and_evaluation.ipynb`.
2. Set `BATCH_NAME`, `MODELS`, `LIMIT_CASES`, and `CACHE_BUDGET_GB`.
3. Run metadata mode first with `METADATA_ONLY = True`.
4. Run inference with `METADATA_ONLY = False` and `ALLOW_DOWNLOADS = True`.
5. Confirm the notebook artifact table shows saved CSV/JSON/Markdown files.
6. Only then set `DELETE_SELECTED_CACHE_AFTER_RUN = True` for temporary
   comparison models.

Heavy model inference should stay in:

```text
notebooks/model_benchmarks/runners/security_model_evaluation.py
```

The notebook should control the script and read the saved outputs. It should
not directly load multiple Transformers models into the Jupyter kernel.
