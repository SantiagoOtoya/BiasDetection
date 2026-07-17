# BIAS Detection Production Classifier Model Card

## Release

- Artifact: `Bias Encoder/models/all-mpnet-base-v2-babe-v3/best`
- Promotion state: `eligible`
- Heads schema: `classification_heads/v3`
- Calibration schema: `calibration/v3`
- Frozen schema: `frozen_artifact/v1`
- Classifier input: `target_sentence_marked/v1`
- Artifact manifest SHA-256:
  `d6eefc2a41f5c4cbeb6cea6043f03e665baf32fd05fda02f9e4ce61f28fa4ce6`
- Frozen artifact SHA-256:
  `df00d6c209b884bbf704eba1a7c990c5e35fcdb480712cae7835284056f5f4dc`
- Model SHA-256:
  `ab9b10f3c1f84488541329cacde2575a4c48ae172661004b796c0d161c70a9d4`
- Heads SHA-256:
  `0933fc0ba048d26a37d6fe97c6d05bfff465b0a4fa1af60e8676b7c340ef3e34`
- Locked scorecard SHA-256:
  `930b70c4c5c7856bc77b505b7c9a09b14948863c6b0813e04f0e8464de99e129`

## Architecture

The model uses `sentence-transformers/all-mpnet-base-v2`. Each normalized target
sentence is serialized as `[TARGET]\n<sentence>` and encoded once. One shared
768-dimensional embedding feeds independent one-hidden-layer scalar MLP heads:

- `not_biased` versus `biased`;
- `objective_style` versus `opinionated_style`.

The first six encoder layers were frozen during four training epochs. Opinion
style is auxiliary and cannot affect bias classification or report selection.

## Data and split

The release uses 4,118 canonical BABE records. No MBIC records were included.
The immutable group-aware split contains:

- train: 2,055;
- development: 619;
- calibration: 620;
- locked test: 824.

The manifest reports no canonical-group or text-hash overlap. Event IDs and
publication dates are unavailable, so event-disjoint and temporal
generalization are not established. Data use and redistribution remain subject
to the original dataset terms.

## Locked-test results

- Bias macro-F1: 0.8202
- Bias ROC-AUC: 0.9143
- Bias PR-AUC: 0.9300
- Opinion-style macro-F1: 0.8300
- Opinion-style bootstrap 95% lower bound: 0.8039
- Clear-bias precision: 0.9698
- Clear-bias precision 95% Wilson lower bound: 0.9488
- Clear-bias recall: 0.6136
- Clear-bias coverage: 0.3617

Every configured promotion gate passed.

## Calibration limitation

Temperature scaling selected identity temperatures of 1.0. The confident
`no_clear_bias` and `objective_style` lower regions failed their calibration
Wilson targets and are disabled. On the locked test, this produces:

- `clear_bias`: 36.2%;
- `possible_bias`: 63.8%;
- `no_clear_bias`: 0%;
- `opinionated_style`: 37.6%;
- `uncertain`: 62.4%;
- `objective_style`: 0%.

This is a deliberate fail-safe: unsupported confident negative decisions fall
back to an uncertain state.

## Intended use

The classifier selects high-precision sentence-level media-bias candidates for
a restrained report. It does not determine truth, intent, source credibility,
or political ideology. Classifier output alone cannot emit factual support or
contradiction.

Known limitations include BABE-only supervision, no temporal or event-disjoint
external evaluation, simple sentence segmentation, and high abstention outside
the confident positive regions.

## Report model

Local report generation uses
`meta-llama/Llama-3.1-8B-Instruct` at revision
`0e9e39f249a16976918f6564b8830bc894c89659`. Model weights are cached in the
user Hugging Face cache and are not committed to this repository. Use is
subject to the Llama 3.1 Community License and Acceptable Use Policy. Retain the
required attribution when distributing Llama materials or an integrated
product.

Local offline validation on 2026-07-12 passed all six provider-free golden
cases. On the RTX 4090, model loading took 6.48 seconds and used a peak of
16.06 GB CUDA memory. The run covered claim extraction, direct support and
contradiction assessment, insufficient-evidence handling, prompt-injection
resistance, and citation guards.

## Recorded environment

- Python 3.13.14
- Torch 2.12.1+cu132
- CUDA runtime 13.2
- NVIDIA GeForce RTX 4090, 24,564 MiB
- NVIDIA driver 610.47
- Transformers 5.13.1
- SentenceTransformers 5.6.0
- Accelerate 1.14.0
- Safetensors 0.8.0
- huggingface_hub 1.23.0
