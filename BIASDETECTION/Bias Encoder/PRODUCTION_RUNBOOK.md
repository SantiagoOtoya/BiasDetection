# Production Classifier Runbook

The three approval-gated production commands below have completed. They are
retained as the immutable release record and must not be rerun against the
existing output paths.

All commands used the CUDA-capable project interpreter at
`..\.venv\Scripts\python.exe`. The system Python remains CPU-only and must not
be used as a silent fallback.

Release result: the locked acceptance gate passed and
`models/all-mpnet-base-v2-babe-v3/best/artifact_manifest.json` is `eligible`.

## 1. Training

```powershell
Set-Location -LiteralPath 'C:\Users\c\Documents\BIAS Detection\Bias Encoder'

& '.\.venv\Scripts\python.exe' .\finetune_all_mpnet_babe.py `
  --model-name 'sentence-transformers/all-mpnet-base-v2' `
  --data-dir .\BABE_HF `
  --classifier-mode production `
  --split-strategy group `
  --group-column news_link `
  --leakage-policy error `
  --validation-size 0.15 `
  --calibration-size 0.15 `
  --test-size 0.20 `
  --split-manifest .\artifacts\production_v3_split_manifest.json `
  --canonical-data-manifest .\artifacts\production_v3_canonical_data_manifest.json `
  --output-dir .\models\all-mpnet-base-v2-babe-v3 `
  --epochs 4 `
  --batch-size 16 `
  --learning-rate 2e-5 `
  --warmup-ratio 0.1 `
  --weight-decay 0.01 `
  --max-seq-length 256 `
  --freeze-first-n-layers 6 `
  --head-hidden-dim 256 `
  --dropout 0.2 `
  --opinion-style-loss-weight 0.3 `
  --class-weighting balanced `
  --checkpoint-selection multi_objective `
  --early-stopping-patience 2 `
  --seed 42 `
  --device cuda `
  --fp16
```

- Inputs: the two `BABE_HF` parquet files and base MPNet artifact.
- Split manifest: created at `artifacts/production_v3_split_manifest.json`.
- Outputs: `models/all-mpnet-base-v2-babe-v3`, its `best` checkpoint, and the
  canonical/split manifests.
- CUDA/runtime: yes; approximately 15–45 minutes on the RTX 4090.
- Overwrite: no current target exists; stop if any output target appears.
- Partitions: all source rows are canonicalized; optimization uses train and
  checkpoint selection uses development. Calibration/locked test are not scored.
- Success: zero leakage, v3 target-only checkpoint, development-selected best
  checkpoint, and no locked-test metrics. Any CPU fallback or test evaluation is
  failure.

## 2. Calibration and thresholds

```powershell
Set-Location -LiteralPath 'C:\Users\c\Documents\BIAS Detection\Bias Encoder'

& '.\.venv\Scripts\python.exe' .\calibrate_sbert_heads.py `
  --sbert-model-dir .\models\all-mpnet-base-v2-babe-v3\best `
  --classification-heads .\models\all-mpnet-base-v2-babe-v3\best\classification_heads.pt `
  --output-file .\models\all-mpnet-base-v2-babe-v3\best\calibration.json `
  --data-dir .\BABE_HF `
  --split-manifest .\artifacts\production_v3_split_manifest.json `
  --canonical-data-manifest .\artifacts\production_v3_canonical_data_manifest.json `
  --artifact-manifest .\models\all-mpnet-base-v2-babe-v3\best\artifact_manifest.json `
  --bias-target-precision 0.90 `
  --bias-target-npv 0.90 `
  --opinion-target-precision 0.90 `
  --opinion-target-npv 0.90 `
  --minimum-predicted-support 30 `
  --precision-confidence 0.95 `
  --batch-size 32 `
  --device cuda
```

- Inputs: best v3 model/heads/artifact manifest plus both data manifests.
- Output: `best/calibration.json`; the artifact manifest is atomically updated.
- CUDA/runtime: yes; approximately 2–5 minutes.
- Overwrite: calibration must not exist; model/heads are never overwritten.
- Partition: only calibration rows enter model scoring, fitting, or threshold
  search. Full canonical data is read only to verify immutable bindings.
- Success: two exactly-once temperature scalers, four deterministic Wilson-LCB
  regions, valid ordering, and `frozen_artifact/v1` with
  `pending_locked_test`. Any other-partition fitting is failure.

## 3. One-shot locked-test evaluation

```powershell
Set-Location -LiteralPath 'C:\Users\c\Documents\BIAS Detection\Bias Encoder'

& '.\.venv\Scripts\python.exe' .\evaluate_locked_test.py `
  --sbert-model-dir .\models\all-mpnet-base-v2-babe-v3\best `
  --calibration-file .\models\all-mpnet-base-v2-babe-v3\best\calibration.json `
  --artifact-manifest .\models\all-mpnet-base-v2-babe-v3\best\artifact_manifest.json `
  --data-dir .\BABE_HF `
  --split-manifest .\artifacts\production_v3_split_manifest.json `
  --canonical-data-manifest .\artifacts\production_v3_canonical_data_manifest.json `
  --gate-config .\promotion_gates.json `
  --output-file .\artifacts\evaluations\all-mpnet-base-v2-babe-v3\locked_test_scorecard.json `
  --evaluation-lock .\artifacts\evaluations\all-mpnet-base-v2-babe-v3\locked_test_once.json `
  --bootstrap-resamples 2000 `
  --batch-size 32 `
  --device cuda `
  --promote-if-gates-pass
```

- Inputs: frozen model, calibration, artifact/data/split manifests, and
  `promotion_gates.json`.
- Outputs: aggregate scorecard, permanent one-shot marker, and an atomically
  updated artifact manifest. No row-level locked predictions are exported.
- CUDA/runtime: yes; approximately 2–5 minutes.
- Overwrite: scorecard/lock must not exist; model/calibration are immutable.
- Partition: locked test only.
- Success: complete binary metrics, Wilson bounds, ROC/PR-AUC, confusion
  matrices, decision-state coverage, and every configured gate passing.
  Passing sets `eligible`; failure sets `rejected` and exits with code 2.

The locked preflight can be run with the same paths plus `--preflight-only`;
it validates the frozen artifact without loading any partition rows or creating
the one-shot marker.

## 4. Production inference

The CLI now defaults to the promoted v3 `best` artifact and pinned Llama
revision. A classifier-only smoke test requires no model download:

```powershell
Set-Location -LiteralPath 'C:\Users\c\Documents\BIAS Detection'

& '.\.venv\Scripts\python.exe' `
  '.\Bias Encoder\LLM-inference\infer_bias_llm.py' `
  --article-text 'The reckless proposal betrayed every hardworking family.' `
  --prompt-only `
  --output-file - `
  --device cuda
```

Local Llama generation uses:

- model `meta-llama/Llama-3.1-8B-Instruct`;
- revision `0e9e39f249a16976918f6564b8830bc894c89659`;
- the normal user Hugging Face cache;
- a locally authenticated read-only Hugging Face token for the gated model.

`--prompt-only` does not load Llama and therefore does not perform claim
extraction, evidence assessment, or report generation.
