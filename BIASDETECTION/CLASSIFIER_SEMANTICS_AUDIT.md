# Classifier Semantics Audit

Audited 2026-07-12 after integration, production training, calibration,
one-shot locked-test evaluation, and conditional promotion.

`CLASSIFIER_SEMANTICS_SPEC.md` is authoritative. `ImplementationPLAN.txt` is a
historical handoff and is superseded where it conflicts with the specification.

## Verification performed

- 117 tests under `Bias Encoder/tests` pass.
- 6 tests under `Backend/backend/tests` pass.
- Production training, calibration, inference, retrieval, evidence, evaluation,
  backend, and API modules pass `python -m py_compile`.
- Strict default-path v3 prompt-only inference loads the promoted artifact and
  its bound calibration on CUDA.
- Offline local Llama validation passes 6/6 provider-free golden cases on the
  pinned revision, including claim extraction, evidence assessment,
  prompt-injection resistance, and citation guarding.

## Authoritative implementation trace

### Model and inputs

Production uses `classification_heads/v3`: one shared encoder pass followed by
independent scalar bias and opinion-style heads. The required input contract is
`target_sentence_marked/v1`, serialized exactly as `[TARGET]\n<sentence>` in
training, calibration, locked evaluation, CLI inference, and backend inference.
Surrounding context is retained only downstream for claims and reports.

The deprecated middle-versus-writer stage is absent from production checkpoints
and outputs. Flat and hierarchical three-tier heads load only in explicit legacy
mode and cannot be promoted.

### Calibration and decisions

`calibration/v3` fits independent temperature scalers from raw calibration
logits. Inference computes `sigmoid(logit / temperature)` once. Four independent
Wilson-LCB threshold regions produce:

- `no_clear_bias | possible_bias | clear_bias`
- `objective_style | uncertain | opinionated_style`

Insufficient support disables the affected confident region. There is no
combined opinion/report threshold.

### Selection and opinion invariance

Default selection uses `clear_bias`; `possible_bias` requires an explicit flag.
Opinion style is not an input to bias decisions, selection, claim extraction,
retrieval routing, or evidence assessment. Tests cover unchanged selection
under objective/opinionated style and confirm that an opinion-only candidate
cannot invoke retrieval.

### Retrieval and evidence

Only selected bias candidates enter claim extraction, and only verifiable claims
enter retrieval. Retrieval implements `off`, `web`, `corpus`, and `hybrid`
transport modes with provenance and safe degradation.

Retrieval items contain no verdict or `supports_claim` field. The neutral source
type is `wire_service`; `wire_factual_domains` is recognized only as a legacy
configuration input alias. Search snippets remain discovery-only.

`evidence_assessment.py` is the sole producer of `supported`, `contradicted`,
`insufficient_evidence`, `not_verifiable`, and `not_assessed`. Strong findings
require an assessment-eligible claim-associated citation. Classifier-only JSON
contains no evidence status or strong finding.

### API

`analysis_response/v2` exposes separate bias, writing-style, retrieval, and
evidence contracts. Public material kinds include `search_snippet`,
`source_excerpt`, and `corpus_chunk`. Evidence citations expose no verdict.

Legacy score/severity, fact-check, and raw label/probability fields remain
nullable and marked deprecated. Production never reverse-maps new states into
old factuality or severity vocabulary. Mock mode fabricates neither evidence nor
legacy fact checks, and invalid analyzer modes fail explicitly.

## Artifact freeze and locked-test policy

Calibration bound the model, heads, input contract, data/split manifests, code,
calibration, and thresholds into `frozen_artifact/v1`. The resulting artifact
passed its one-shot locked evaluation and now has `promotion_state=eligible`.

`evaluate_locked_test.py` is the sole production locked-test entrypoint. It:

1. validates the complete frozen binding and configured split paths;
2. creates `locked_test_once/v1` atomically before loading locked labels;
3. loads only the `locked_test` partition for scoring;
4. writes aggregate `evaluation_scorecard/v2` metrics without row predictions;
5. binds the scorecard hash and sets `eligible` or `rejected` atomically;
6. refuses repeat evaluation for the same frozen artifact.

Promotion gates are versioned in `Bias Encoder/promotion_gates.json`:

- clear-bias precision one-sided 95% Wilson LCB at least 0.90;
- binary opinion-style article-bootstrap macro-F1 LCB at least 0.65;
- objective and opinionated recall Wilson LCBs at least 0.50;
- minimum evaluable support 30.

The scorecard also reports binary accuracy, balanced accuracy, per-class
precision/recall/F1 and Wilson bounds, confusion matrices, ROC-AUC, PR-AUC, and
all bias/style decision-state counts and percentages.

## Legacy and disconnected inventory

- Existing deployed heads are untyped two-logit bias plus flat three-class
  opinion and remain diagnostic-only.
- Existing schema-1 calibration is unbound and validation-fitted. Strict mode
  rejects it; explicit legacy mode may inspect it.
- `Backend/backend/fact_checker.py` and `scoring.py` retain old terminology but
  are disconnected from production and mock responses.
- The extension UI still renders old score/severity and fact-check concepts and
  has not been migrated in this integration task.
- `Backend/extension` remains a disconnected incomplete copy.

## Current release status

- The promoted artifact is
  `Bias Encoder/models/all-mpnet-base-v2-babe-v3/best`.
- Its locked-test acceptance gate passed and its manifest is `eligible`.
- The project `.venv` has CUDA-enabled Torch 2.12.1+cu132 and sees the RTX 4090.
- The lower `no_clear_bias` and `objective_style` calibrated regions are safely
  disabled because their calibration Wilson bounds did not meet target. This
  artifact therefore falls back to `possible_bias` and `uncertain` for those
  regions.
- Live provider integration and deployment controls remain separate work.

The exact approval-gated commands, paths, overwrite rules, partitions, runtimes,
and success criteria are in `Bias Encoder/PRODUCTION_RUNBOOK.md`.
