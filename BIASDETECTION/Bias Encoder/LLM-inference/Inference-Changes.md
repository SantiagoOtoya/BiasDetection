# Production Inference Integration

This document records the current SBERT-to-Llama inference path. Historical
three-class opinion and legacy argmax behavior are available only through
explicit compatibility modes.

## Production defaults

- Classifier: `models/all-mpnet-base-v2-babe-v3/best`
- Heads: `classification_heads/v3`
- Calibration: `calibration/v3`
- Input contract: `target_sentence_marked/v1`
- Selection: calibrated `clear_bias` only by default
- LLM: `meta-llama/Llama-3.1-8B-Instruct`
- LLM revision: `0e9e39f249a16976918f6564b8830bc894c89659`
- Context: two sentences before and after each selection
- Output: `LLM-inference/outputs/bias_llm_results.jsonl`

The shared MPNet embedding feeds independent scalar bias and opinion-style
heads. Opinion style is auxiliary and cannot affect bias classification,
selection, claim extraction, retrieval, or evidence assessment.

## Flow

1. Load the promoted, hash-bound v3 classifier and calibration.
2. Split the article and serialize each target as `[TARGET]\n<sentence>`.
3. Encode each target once and run both independent heads.
4. Select `clear_bias`; include `possible_bias` only when explicitly requested.
5. Merge overlapping context windows.
6. When Llama is loaded, extract verifiable claims from selected sentences.
7. Optionally retrieve evidence for those claims.
8. Assess claims only against assessment-eligible excerpts.
9. Generate a restrained prose report with validated citation IDs.

`--prompt-only` stops before Llama loading. It produces classifier decisions and
report prompts, but claim extraction and evidence assessment remain
`not_assessed` and retrieval is not invoked because there are no extracted
claims.

## Local Llama validation

The pinned snapshot is cached locally and was verified offline on CUDA. The
provider-free golden suite passes 6/6 cases: claim extraction, directly
supported and contradicted excerpts, insufficient evidence, prompt-injection
resistance, and strong-status citation guards. The output is retained in
`outputs/v3_llama_golden.jsonl`.

## Production boundaries

- Search snippets are discovery-only and cannot produce strong evidence states.
- Retrieval rank and source reputation are not evidence verdicts.
- Only the evidence-assessment stage may emit `supported` or `contradicted`.
- Missing or failed evidence must never produce fabricated citations.
- PDF parsing, article extraction, unrestricted scraping, backend deployment,
  and UI rendering are outside this entrypoint.
