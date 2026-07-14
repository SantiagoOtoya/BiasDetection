# Implementation Status — BIAS Detection

Updated 2026-07-12 after production training, calibration, locked evaluation,
promotion, inference-default correction, and local Llama validation.

`CLASSIFIER_SEMANTICS_SPEC.md` remains authoritative.

## Production classifier

- Production training completed with `classification_heads/v3`: one shared
  MPNet encoder result feeding independent scalar bias and opinion-style heads.
- The immutable group split contains 2,055 train, 619 development, 620
  calibration, and 824 locked-test rows with no reported group or text-hash
  overlap.
- `calibration/v3` is bound to the model, heads, data/split manifests, code,
  input contract, calibrators, and thresholds.
- The one-shot locked evaluation completed. Every configured release gate
  passed and `models/all-mpnet-base-v2-babe-v3/best` is `eligible`.
- Clear-bias locked-test precision is 0.9698, with a one-sided 95% Wilson lower
  bound of 0.9488 and recall of 0.6136.
- The conservative lower regions remain disabled: production emits no
  `no_clear_bias` or `objective_style` decisions for this artifact and falls
  back to `possible_bias` or `uncertain` instead.

## Runtime

- The project `.venv` uses Python 3.13.14, Torch 2.12.1+cu132, and CUDA 13.2.
- CUDA is available on the NVIDIA GeForce RTX 4090.
- CLI inference defaults to the promoted v3 `best` artifact and strict
  calibrated selection. Legacy models require an explicit compatibility mode.
- Llama inference is pinned to
  `meta-llama/Llama-3.1-8B-Instruct` revision
  `0e9e39f249a16976918f6564b8830bc894c89659`.
- The pinned Llama snapshot is cached locally and successfully resolves with
  `HF_HUB_OFFLINE=1`.
- Prompt-only inference validates classification and prompt construction but
  intentionally performs no LLM claim extraction, evidence assessment, or
  report generation.

## Verification

- All 117 `Bias Encoder` unit tests pass.
- Strict default-path v3 prompt-only inference passes on CUDA.
- Full local Llama inference passes on CUDA with calibrated selection, claim
  extraction, and report generation; retrieval remains explicitly off.
- The provider-free Llama golden suite passes 6/6 cases offline, including
  claim extraction, support/contradiction assessment, prompt-injection
  resistance, and citation guarding.
- Model, calibration, data, split, and promotion bindings validate at load.
- Provider-independent retrieval and evidence contracts are unit tested.

## Canonical artifacts

- Model: `Bias Encoder/models/all-mpnet-base-v2-babe-v3/best`
- Calibration: `best/calibration.json`
- Artifact manifest: `best/artifact_manifest.json`
- Locked scorecard:
  `Bias Encoder/artifacts/evaluations/all-mpnet-base-v2-babe-v3/locked_test_scorecard.json`
- Locked marker:
  `Bias Encoder/artifacts/evaluations/all-mpnet-base-v2-babe-v3/locked_test_once.json`

The parent v3 directory contains the final-epoch training export and its
historical `pending_calibration` manifest. It is not the production inference
target; `best` is the sole promoted artifact.

## Separate remaining work

Live Brave/Qdrant provider readiness, corpus construction, backend deployment,
and UI migration remain separate tasks. They are not part of the classifier or
local Llama release described here.
