# Bias Detection Backend

FastAPI server that powers the Chrome extension. It wraps the existing,
unmodified inference code (`../LLM-inference/infer_bias_llm.py`, which imports
`../finetune_all_mpnet_babe.py`) and adds an HTTP API, the bias score, and the
conservative fact-checker.

## Endpoints

- `GET /health` — readiness + current mode (`sbert_loaded`, `llm_loaded`).
- `POST /analyze` — body `{ "text": "...", "title": "...", "url": "..." }` →
  `AnalyzeResponse` (score, level, report, fact_checks, selected_sentences).

## Modes (`ANALYZER_MODE` env var)

| Mode          | SBERT | Llama report | Fact-check        | Needs |
|---------------|-------|--------------|-------------------|-------|
| `mock`        | no    | placeholder  | placeholder       | nothing (default) |
| `prompt-only` | yes   | extractive summary | all `unverified` | CPU, inference deps |
| `cpu`         | yes   | yes (slow)   | yes               | CPU, ~16 GB Llama download |
| `gpu`         | yes   | yes          | yes               | CUDA GPU |

`mock` lets you test the extension wiring with no ML stack installed.
`prompt-only` runs the real fine-tuned SBERT classifier + scoring on CPU without
loading Llama — ideal for local testing on a machine without a GPU.

## Install

```powershell
# Server only (enough for mock mode)
python -m pip install -r requirements-server.txt

# Real analysis (SBERT / Llama) also needs the inference deps
python -m pip install -r ..\LLM-inference\requirements-inference.txt
```

## Run

```powershell
# Mock (no models)
$env:ANALYZER_MODE = "mock"
python -m uvicorn server:app --host 0.0.0.0 --port 8000

# Real SBERT on CPU, no Llama (local testing)
$env:ANALYZER_MODE = "prompt-only"
python -m uvicorn server:app --host 0.0.0.0 --port 8000

# Full pipeline on the CUDA machine
$env:ANALYZER_MODE = "gpu"
$env:HF_TOKEN = "hf_xxx"   # Llama 3.1 8B is gated on Hugging Face
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` makes the server reachable from your other machine over the LAN
(point the extension's Backend URL at `http://<gpu-machine-ip>:8000`) or via a
tunnel.

## Model assembly

On first real run, `model_loader.py` builds a loadable SentenceTransformer
checkpoint under `backend/models/assembled-sbert-babe/`:

1. Downloads only the scaffolding (config/tokenizer/pooling) from base
   `sentence-transformers/all-mpnet-base-v2`.
2. Overwrites the transformer weights with the fine-tuned `model.safetensors`
   from `BiLSTM/BIAS_Detection`.
3. Downloads `classification_heads.pt` (the bias + opinion heads).

The base model supplies **only** the missing scaffolding. The encoder weights are
verified by SHA-256 against the downloaded fine-tuned file, so the friend's
trained model is what actually runs. Delete the folder to force a rebuild.
