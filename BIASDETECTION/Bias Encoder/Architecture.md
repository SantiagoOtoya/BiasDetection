# Production Classifier Architecture

`CLASSIFIER_SEMANTICS_SPEC.md` is authoritative. Production uses one shared
`all-mpnet-base-v2` encoder and two independent scalar heads:

```text
input = "[TARGET]\n" + normalized_sentence
embedding = encoder(input)                  # exactly once
bias_logit = bias_head(embedding)
opinionated_style_logit = style_head(embedding)
```

The input contract is `target_sentence_marked/v1`. Surrounding article context
is retained for claim extraction, evidence retrieval, assessment, and reports,
but is not part of classifier input. Training, calibration, locked evaluation,
CLI inference, and backend inference all use the same serializer.

Production checkpoints use `classification_heads/v3` with binary
`not_biased/biased` and `objective_style/opinionated_style` mappings. The old
flat and hierarchical opinion heads are explicit legacy-only paths.

## Data and checkpoint selection

Group-aware preparation creates immutable train, development, calibration, and
locked-test partitions. Article/event/text-hash overlap is an error. Training
uses independent masked BCE-with-logits losses; missing supervision for one
task does not discard the other. The best checkpoint is chosen from development
metrics only. The training command must omit `--evaluate-test-final`.

## Calibration and decisions

`calibration/v3` fits separate temperature scalers from raw calibration logits
and applies `sigmoid(logit / temperature)` exactly once. Four Wilson-LCB
threshold searches produce:

- `no_clear_bias | possible_bias | clear_bias`
- `objective_style | uncertain | opinionated_style`

Each confident region requires configured precision/NPV and minimum support.
Unsupported regions are disabled and route to the middle state. Opinion style
never affects bias, report eligibility, claim extraction, or retrieval.

Successful calibration binds the model, heads, data/split manifests, code,
input contract, calibrators, and thresholds into `frozen_artifact/v1` with
`promotion_state=pending_locked_test`.

## Locked evaluation and promotion

`evaluate_locked_test.py` is the only production locked-test command. It creates
an exclusive `locked_test_once/v1` marker before loading locked rows, writes only
aggregate `evaluation_scorecard/v2` output, and refuses repeat evaluation.
Promotion uses `promotion_gates.json`; passing sets the frozen artifact to
`eligible`, while any failed gate sets it to `rejected`.

See `PRODUCTION_RUNBOOK.md` for the exact approval-gated commands.
