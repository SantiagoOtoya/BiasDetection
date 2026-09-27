# Bias Detection Backend

FastAPI server that powers the Chrome extension. It wraps the existing,
unmodified inference code (`../infer_bias_llm.py`, which imports
`../finetune_all_mpnet_babe.py` and `../evidence_retrieval.py`) and adds an HTTP
API, the bias score, calibrated-selection metadata, and the conservative
fact-checker.

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
python -m pip install -r ..\requirements-inference.txt
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

## Model stacks (`MODEL_STACK` env var)

Two inference stacks coexist in the repo and share module names, so exactly one
is loaded per process (enforced by `stack_loader.py`):

| `MODEL_STACK` | Stack | Model | Selection |
|---------------|-------|-------|-----------|
| `v2` (default) | repo-root modules | assembled HF v2 encoder + 4-class heads | argmax / threshold |
| `v3` | `BIASDETECTION/Bias Encoder` handoff | v3 encoder (auto-downloaded from `BiLSTM/BIASDETECTION_`) + scalar heads | strict calibrated: `clear_bias` / `possible_bias` / `no_clear_bias` |

v3 notes:
- First run downloads `model.safetensors` (~438 MB) into the handoff model dir;
  the strict loader then verifies the manifest's directory-composite SHA-256 and
  **refuses to run** on any mismatch — the wrong model can never load silently.
- Only `clear_bias` sentences are selected by default (locked-test precision
  0.97). Set `INCLUDE_POSSIBLE_BIAS=true` to also include `possible_bias`.
- Responses add `bias_assessment`, `opinion_style`, `uncertainty_status` per
  selected sentence and `meta.stack: "v3"`.
- Evidence (web mode via Brave) needs `ENABLE_EVIDENCE=true` plus
  `BRAVE_SEARCH_API_KEY` and `CORPUS_FETCH_CONTACT`; claim assessments map to
  the fact list as supported→verified, contradicted→disputed,
  insufficient→unverified (never "false"). Corpus/Qdrant mode is not wired yet.
- The `BIASDETECTION/**` handoff is byte-exact: a root `.gitattributes` marks it
  `-text` so git never rewrites line endings (the handoff's SHA-256 manifests
  hash raw bytes).

## Selection mode (calibrated abstention)

Sentence selection reuses the inference module's calibrated confidence gate:

| `SELECTION_MODE` | Behavior |
|------------------|----------|
| `auto` (default) | Use `calibration.json` in the model dir if present, else legacy argmax |
| `argmax`         | Force the legacy `Biased` / non-factual argmax rule |
| `calibrated`     | Require `calibration.json` (error if missing) |

Generate `calibration.json` with `python ..\calibrate_sbert_heads.py
--sbert-model-dir <model_dir> --target-precision 0.90`. Each response's
`meta.selection_policy` and `meta.abstention_summary` report which rule ran and
why sentences were abstained; per-sentence `selection_reasons` /
`abstention_reasons` are included on `selected_sentences`.

## Trusted evidence (optional, off by default)

Evidence retrieval is disabled unless `ENABLE_EVIDENCE=true`. When enabled it
requires `BRAVE_SEARCH_API_KEY` and only accepts strictly trusted sources
(official/primary, empirical research, AP/Reuters — see `../evidence_retrieval.py`).

| Env var | Default | Meaning |
|---------|---------|---------|
| `ENABLE_EVIDENCE` | `false` | Turn evidence retrieval on/off |
| `BRAVE_SEARCH_API_KEY` | — | Required when evidence is enabled |
| `EVIDENCE_PROVIDER` | `brave` | Search provider |
| `MAX_EVIDENCE_ITEMS` | `5` | Cap on trusted items per article |
| `EVIDENCE_TIMEOUT_SECONDS` | `10` | Per-request search timeout |

Fact-checker reconciliation: with evidence **off** (or no key / none found), the
conservative context-only checker runs and every claim defaults to `unverified`.
With evidence **on** and trusted items found, an evidence-grounded review runs
that may reach `verified` / `disputed` / `false` and cites evidence ids, still
defaulting to `unverified` when the evidence is insufficient.

```powershell
$env:ANALYZER_MODE = "gpu"
$env:ENABLE_EVIDENCE = "true"
$env:BRAVE_SEARCH_API_KEY = "..."
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

## Relevance filter (page-furniture removal)

Before SBERT classification, `relevance_filter.py` removes sentences that are
not part of the article's main content. Two conservative layers:

1. **Heuristics** — reasons `too_short` (UI fragments like "Menu"),
   `boilerplate_keyword` (newsletter/cookie/share/related/comment prompts),
   `duplicate` (repeated sentences).
2. **Semantic** — embeds a topic anchor (the extension's `lead_text`: headline +
   first paragraph) plus each sentence with the already-loaded fine-tuned SBERT
   and removes sentences below the cosine threshold — but **only** if they also
   look UI-like. Long-form prose is never removed semantically, so genuine
   article content (quotes, background, tangents) survives. Reason:
   `low_topic_similarity`.

| Env var | Default | Meaning |
|---------|---------|---------|
| `ENABLE_RELEVANCE` | `true` | Turn the whole filter on/off |
| `ENABLE_SEMANTIC_RELEVANCE` | `true` | Layer 2 on/off (heuristics stay on) |
| `RELEVANCE_THRESHOLD` | `0.18` | Cosine similarity floor for UI-like sentences |

The overall bias score and the "X of Y relevant sentence(s)" caption count only
sentences that pass the filter. Debug counts are returned in
`meta.relevance`: `total_sentences`, `sentences_after_relevance_filter`,
`sentences_removed_as_irrelevant`, `removed_reason_counts`, `semantic_ran` —
the extension popup shows them in its footer and logs details to the console.

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
