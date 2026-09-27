# Bias Detection

A Chrome extension that detects biased and opinionated language in news, journals,
and papers, then explains it with a grounded report, a bias-intensity score, and a
conservative fact-check list. Because a fine-tuned SBERT classifier and an 8B LLM
cannot run in a browser, the extension talks over HTTP to a local/remote Python
backend that keeps the models loaded in memory.

## Two model stacks

The backend can run either of two classifier generations, selected at startup with
the `MODEL_STACK` environment variable. Exactly one is loaded per process
(`backend/stack_loader.py` enforces this — the stacks share module names).

| Stack | `MODEL_STACK` | Encoder | Selection |
|-------|---------------|---------|-----------|
| **v2** (default) | `v2` | root inference files + HF v2 weights | argmax / threshold |
| **v3** | `v3` | `BIASDETECTION/` handoff + strict-verified v3 weights | calibrated `clear_bias` / `possible_bias` / `no_clear_bias` |

v2 stays the default until v3 is validated end-to-end. v3 performs strict SHA-256
artifact verification at load and refuses to run on any mismatch.

## Repository layout

```
BiasDetection/
├─ README.md                     ← you are here
├─ backend/                      ← FastAPI server (the app). See backend/README.md
│   ├─ server.py, pipeline.py, real_pipeline.py   (shared + v2)
│   ├─ stack_loader.py           (one-stack-per-process isolation)
│   ├─ v3_adapter.py, v3_mapping.py               (v3 path)
│   ├─ model_loader.py, scoring.py, schemas.py, relevance_filter.py, fact_checker.py
│   └─ README.md                 ← run guide, MODEL_STACK, evidence, relevance knobs
├─ extension/                    ← Manifest V3 Chrome extension (popup, scraper, options)
├─ tests/                        ← unit tests for both stacks (mock-based; no weights)
├─ docs/                         ← v2 design notes (v2-Architecture, v2-Inference-Changes, …)
│
│   # v2 inference stack (repo root — loaded when MODEL_STACK=v2)
├─ infer_bias_llm.py, evidence_retrieval.py, finetune_all_mpnet_babe.py
├─ calibrate_sbert_heads.py, requirements-inference.txt, requirements-finetune.txt
│
└─ BIASDETECTION/                ← READ-ONLY v3 handoff (do not edit — see below)
    ├─ HANDOFF_README.md, MODEL_CARD.md, SHA256SUMS.txt
    └─ Bias Encoder/…            ← v3 training/inference/eval + model artifacts
```

## `BIASDETECTION/` is read-only

It is a sanitized, byte-exact production handoff whose SHA-256 manifests hash raw
file bytes. **Do not edit, reformat, or re-encode anything under it** — v3's strict
artifact binding will reject any change. A root `.gitattributes` marks
`BIASDETECTION/** -text` so git never rewrites its line endings (without that,
CRLF conversion breaks the hash verification on checkout).

The v3 encoder weights (`model.safetensors`, ~438 MB) are **not** committed; the
backend downloads them from `BiLSTM/BIASDETECTION_` into the handoff model dir on
first `MODEL_STACK=v3` run and verifies them before serving.

## Quick start (v2, local CPU — no GPU needed)

```powershell
cd backend
python -m pip install -r requirements-server.txt
python -m pip install -r ..\requirements-inference.txt
$env:ANALYZER_MODE = "prompt-only"   # real SBERT classification, no Llama
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Then load `extension/` via `chrome://extensions` → Developer mode → Load unpacked,
open an article, and click **Run analysis**. Use the ⚙ options page to point the
Backend URL at a GPU machine (LAN IP / tunnel) for the full report.

The full pipeline (Llama 3.1 8B report + evidence) requires a CUDA GPU with ~16 GB
VRAM — run the backend with `ANALYZER_MODE=gpu` (and `MODEL_STACK=v3` for the
production classifier) on that machine. See [backend/README.md](backend/README.md).

## Models

- Fine-tuned SBERT + MLP heads (v3): https://huggingface.co/BiLSTM/BIASDETECTION_
- Report LLM (gated): https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
