# SBERT to Llama Inference

`LLM-inference/infer_bias_llm.py` loads the fine-tuned SBERT checkpoint and the two saved
classification heads, classifies article sentences, selects biased or
opinionated sentences, expands each selected region with surrounding article
context, and sends those context windows to `meta-llama/Llama-3.1-8B-Instruct`.

The SBERT sentence embeddings are internal to classification. Llama receives
only the selected sentence text and surrounding article context.

## Defaults

- SBERT checkpoint: `models/all-mpnet-base-v2-babe/best`
- Classification heads: `classification_heads.pt` in the SBERT checkpoint
- LLM: `meta-llama/Llama-3.1-8B-Instruct`
- Sentence selection: predicted `Biased` or opinion label other than
  `Entirely factual`
- Context window: two sentences before and after each selected sentence
- Output: `LLM-inference/outputs/bias_llm_results.jsonl`

## Install

```powershell
python -m pip install -r .\LLM-inference\requirements-inference.txt
```

Local Llama 3.1 8B inference is intended for CUDA. The machine may have an
NVIDIA GPU available, but Python still needs a CUDA-enabled Torch install.
Use `--prompt-only` to validate SBERT classification and prompt generation
without loading Llama.

The Llama model is gated on Hugging Face. Authenticate with the Hugging Face CLI
or set `HF_TOKEN`.

## Examples

Analyze raw article text and write LLM-ready prompts without loading Llama:

```powershell
python .\LLM-inference\infer_bias_llm.py `
  --article-text "School systems are adopting BLM curriculum at an alarming rate, indoctrinating children to achieve Marxist objectives." `
  --prompt-only `
  --output-file .\LLM-inference\outputs\smoke.jsonl
```

Analyze a text file and run local Llama generation:

```powershell
python .\LLM-inference\infer_bias_llm.py `
  --article-file .\article.txt `
  --output-file .\LLM-inference\outputs\article_report.jsonl
```

Analyze a dataset-style CSV with full article text:

```powershell
python .\LLM-inference\infer_bias_llm.py `
  --input-file .\data\final_labels_MBIC.csv `
  --article-column article `
  --id-column news_link `
  --prompt-only `
  --output-file .\LLM-inference\outputs\mbic_prompts.jsonl
```

## Output

Each JSONL record contains:

- article metadata
- selected biased/opinionated sentence predictions
- merged context windows
- the exact prompt sent to Llama
- the generated Llama report text

The LLM report is intentionally text-grounded, chronological, and written as a
cohesive user-facing digest rather than JSON. It references specific word choice
and misaligned statements, explains what the supplied passage objectively
establishes, and must not expose labels, probabilities, model names, file paths,
prompt mechanics, or other pipeline details. It can describe biased,
misleading, opinionated, or unsupported wording, but it should not call a claim
objectively false unless that is established by the supplied article context.

PDF parsing, web scraping, and extraction are intentionally outside this script.
Those components can feed extracted article text into `--article-text`,
`--article-file`, or `--input-file` when they are ready.
