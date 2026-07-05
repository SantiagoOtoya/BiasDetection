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
- Sentence selection: calibrated confidence gate when `calibration.json` is
  available; otherwise the legacy argmax rule is used
- Context window: two sentences before and after each selected sentence
- Output: `LLM-inference/outputs/bias_llm_results.jsonl`

## Calibration

Run calibration after training to fit temperature scaling and high-precision
selection thresholds from the validation split:

```powershell
python .\calibrate_sbert_heads.py `
  --sbert-model-dir .\models\all-mpnet-base-v2-babe\best `
  --target-precision 0.90
```

Inference defaults to `--selection-mode auto`, which loads
`<sbert-model-dir>/calibration.json` when present. Use `--selection-mode argmax`
to force the legacy rule, or `--selection-mode calibrated` to require a
calibration file.

## Trusted Evidence Support

Evidence retrieval is optional and deliberately narrow. When `--enable-evidence`
is used, the pipeline only accepts evidence from:

- official or primary factual sources such as `.gov`, `.mil`, major public
  agencies, courts, regulators, WHO, UN, OECD, IMF, and World Bank
- empirical research sources such as DOI, PubMed/PMC, arXiv, NBER, SSRN, and
  major academic journal or publisher domains
- high-factual wire sources: AP and Reuters

Other search results are discarded before they can enter the LLM prompt. This
is evidence-supported claim review, not full automated fact-checking. Reports
should describe whether claims are supported, contradicted, or not established
by retrieved trusted evidence.

```powershell
$env:BRAVE_SEARCH_API_KEY = "..."
python .\LLM-inference\infer_bias_llm.py `
  --article-file .\article.txt `
  --enable-evidence `
  --prompt-only `
  --output-file .\LLM-inference\outputs\article_with_evidence.jsonl
```

## Install

```powershell
python -m pip install -r .\LLM-inference\requirements-inference.txt
```
## Examples

Analyze raw article text and write LLM-ready prompts without loading Llama:

```powershell
python .\LLM-inference\infer_bias_llm.py `
  --article-text "School systems are adopting BLM curriculum at an alarming rate, indoctrinating children to achieve Marxist objectives." `
  --selection-mode auto `
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
- selection policy and abstention summary
- selected biased/opinionated sentence predictions
- merged context windows
- optional trusted evidence status and citation metadata per context window
- the exact prompt sent to Llama
- the generated Llama report text

The LLM report is intentionally text-grounded, chronological, and written as a
cohesive user-facing digest rather than JSON. It references specific word choice
and misaligned statements, explains what the supplied passage objectively
establishes, and must not expose labels, probabilities, model names, file paths,
prompt mechanics, or other pipeline details. It can describe biased,
misleading, opinionated, or unsupported wording, but it should not call a claim
objectively false unless that is established by the supplied article context or
trusted retrieved evidence.

PDF parsing, web scraping, and extraction are intentionally outside this script.
Those components can feed extracted article text into `--article-text`,
`--article-file`, or `--input-file` when they are ready.
