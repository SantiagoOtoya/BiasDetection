# SBERT-to-Llama Inference Integration

This document describes the SBERT-to-Llama inference path. The pipeline keeps
the fine-tuned SBERT sentence embedding classifier intact, then uses its
sentence-level bias and opinion predictions to decide what article context
should be sent to an open source LLM.

## Goal

The fine-tuned SBERT model produces sentence embeddings that feed two
non-linear classification heads:

- a primary bias classification head
- an auxiliary opinion classification head

At inference time, the pipeline uses those heads to identify sentences that are
semantically biased or opinionated. It then sends the selected sentence text
plus surrounding article context to `meta-llama/Llama-3.1-8B-Instruct`.

The LLM receives text context, not raw embeddings. Embeddings remain internal to
the SBERT classification step.

## Components

### `LLM-inference/infer_bias_llm.py`

This inference entrypoint loads the saved fine-tuned SBERT checkpoint and the
saved MLP classification heads, classifies article sentences, builds context
windows around selected sentences, and either:

- writes LLM-ready prompts with `--prompt-only`, or
- loads `meta-llama/Llama-3.1-8B-Instruct` locally and generates a structured
  report.

Supported input modes:

- `--article-text "..."`
- `--article-file path.txt`
- `--input-file rows.csv --article-column article`

The script defaults to:

- SBERT checkpoint: `models/all-mpnet-base-v2-babe/best`
- classification heads: `classification_heads.pt` inside that checkpoint
- LLM model: `meta-llama/Llama-3.1-8B-Instruct`
- context window: two sentences before and after each selected sentence
- output file: `LLM-inference/outputs/bias_llm_results.jsonl`

### `LLM-inference/requirements-inference.txt`

This separates inference dependencies from training dependencies. It includes
the packages needed for SBERT inference, table input, and Hugging Face Llama
generation:

- `sentence-transformers`
- `pandas`
- `pyarrow`
- `torch`
- `transformers`
- `accelerate`
- `safetensors`

### `LLM-inference/LLM-Inference.md`

This file provides the runnable inference workflow. It explains:

- the default SBERT and Llama models
- how sentence selection works
- how context windows are built
- how to run prompt-only inference
- how to run local Llama generation
- why PDF parsing and web scraping are outside this script

## Inference Flow

1. Load the fine-tuned SBERT checkpoint.
2. Load `classification_heads.pt`.
3. Reconstruct both MLP heads from checkpoint metadata.
4. Split article text into sentences.
5. Generate SBERT sentence embeddings.
6. Run both classification heads over each embedding.
7. Select a sentence if:
   - the bias head predicts `Biased`, or
   - the opinion head predicts anything other than `Entirely factual`.
8. Build context windows around selected sentences.
9. Merge overlapping windows so nearby selected sentences share context.
10. Build a concise LLM prompt for each context window.
11. Either write the prompt-only JSONL output or call Llama and store the report.

## Output Shape

The JSONL output contains one record per article. Each record includes:

- `article_id`
- input metadata
- SBERT model directory
- Llama model id
- sentence count
- selected sentence count
- selected sentence predictions
- context windows
- prompt text
- optional LLM report

The selected sentence predictions include:

- sentence index
- sentence text
- bias label and probabilities
- opinion label and probabilities

## LLM Report Boundary

The LLM prompt asks for an objective, chronological prose report written as a
cohesive user-facing digest rather than JSON. The report should scale with the
amount and magnitude of bias in the passage and should identify:

- specific biased or opinionated word choice
- misaligned statements where the article's framing outruns the evidence
- how the wording frames the subject, event, person, group, or claim
- what the article context objectively establishes
- what the article context does not establish
- the most objective truth available from the supplied passage
- neutral alternatives when they help clarify the supported claim

The report is text-grounded and user-facing. It analyzes only the article
passage at hand and does not expose labels, probabilities, model names, file
paths, prompt mechanics, or other pipeline details. It does not claim that
something is objectively false unless the provided article context establishes
that. External fact-checking, retrieval, and web search are handled outside this
module.

The SBERT training script remains functional. The inference path imports
reusable helpers from `finetune_all_mpnet_babe.py` instead of modifying the
training flow.

The inference path does not include:

- PDF parsing
- web scraping
- article extraction
- external fact-check retrieval
- embedding persistence

Those components can be connected later by feeding extracted article text into
`infer_bias_llm.py`.
