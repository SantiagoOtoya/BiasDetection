# Classifier, Retrieval, Evidence, and Report Inference

Trusted Brave/source extraction, Qdrant Cloud lifecycle, manifests, backups,
and the retrieval-only benchmark are documented in
[`CORPUS-OPERATIONS.md`](CORPUS-OPERATIONS.md). Provider configuration is loaded
only with the explicit `--env-file` option.

Production inference loads a promoted `classification_heads/v3` checkpoint and
its bound `calibration/v3` sidecar. Missing, legacy, unbound, or unpromoted
artifacts fail explicitly; legacy behavior requires an explicit compatibility
mode.

The CLI production defaults are:

- classifier `../models/all-mpnet-base-v2-babe-v3/best`;
- Llama `meta-llama/Llama-3.1-8B-Instruct`;
- Llama revision `0e9e39f249a16976918f6564b8830bc894c89659`;
- calibrated selection with `clear_bias` report eligibility.

Each article sentence is serialized with `target_sentence_marked/v1` and encoded
once. The shared embedding feeds independent bias and writing-style heads.
Calibrated bias state alone controls selection: `clear_bias` is selected,
`possible_bias` is optional, and `no_clear_bias` is excluded. Opinion style is
auxiliary and cannot change selection or retrieval.

For selected bias candidates, structured claim extraction determines
verifiability. Only verifiable claims reach retrieval. Supported modes are
`off`, `web`, `corpus`, and `hybrid`:

- Brave web results are discovery snippets and cannot alone support strong
  evidence findings.
- Up to three registry-approved source pages may yield assessment-eligible
  `source_excerpt` chunks after controlled extraction.
- Qdrant corpus chunks may be assessment-eligible when their provenance and
  claim association validate.
- Hybrid mode combines provider rankings deterministically and degrades safely.

Retrieval output contains transport status and ranking provenance only.
`EvidenceItem` has no `supports_claim` or verdict field. Wire services use the
neutral `wire_service` source type; `wire_factual_domains` is accepted only as
an old configuration alias.

`evidence_assessment.py` is the sole producer of `supported`, `contradicted`,
`insufficient_evidence`, `not_verifiable`, and `not_assessed`. Strong findings
require a valid assessment-eligible citation. Retrieval similarity, source
reputation, classifier output, and article repetition cannot produce them.

Reports receive calibrated states, surrounding context, extracted claims,
completed assessments, sanitized citations, and artifact provenance. They must
not infer factual truth from objective style or absence of a bias selection.

## Optional providers

Web/source retrieval requires `BRAVE_SEARCH_API_KEY` and
`CORPUS_FETCH_CONTACT`. Corpus and hybrid modes require `QDRANT_URL` and
`QDRANT_API_KEY`. Provider values are loaded from a dotenv file only when
`--env-file` is explicit; process variables win. The legacy local-path config
object remains import-compatible for tests and existing callers, but lifecycle
mutations require an HTTPS Qdrant Cloud URL. The v2 source registry separately
controls discovery, fetching, ingestion, redistribution, rights review, and
retention.

Provider absence or failure returns `none_found`, `partial`, or `error` without
fabricated evidence.

## Local commands

From the repository root, classifier-only prompt construction is:

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\Bias Encoder\LLM-inference\infer_bias_llm.py' `
  --article-text 'The reckless proposal betrayed every hardworking family.' `
  --prompt-only `
  --output-file - `
  --device cuda
```

`--prompt-only` does not run claim extraction, retrieval, evidence assessment,
or report generation because no Llama generator is loaded.

The gated Llama snapshot is cached outside the repository with:

```powershell
& '.\.venv\Scripts\hf.exe' download `
  'meta-llama/Llama-3.1-8B-Instruct' `
  --revision '0e9e39f249a16976918f6564b8830bc894c89659' `
  --exclude 'original/*'
```

Authenticate locally with `hf auth login`; never place a token in repository
files. After the snapshot is cached, `HF_HUB_OFFLINE=1` can be used to verify
that generation requires no network access.

## Commands

Training, calibration, and the one-shot locked evaluation are documented in
`../PRODUCTION_RUNBOOK.md`. Those commands are never invoked by inference.
