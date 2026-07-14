# Trusted retrieval and corpus operations

The authoritative collection contract is fixed at `trusted_corpus_v1`, named
`content` vectors, cosine distance, `sentence-transformers/all-MiniLM-L6-v2`,
the `chunk_text` payload field, and 350-word windows with 50-word overlap.
Qdrant Cloud is authoritative; lifecycle commands do not provision local
Qdrant.

## Configuration

Install the optional provider dependencies:

```powershell
python -m pip install -r requirements-qdrant.txt
```

Copy the repository `.env.example` to `.env.local` and fill in
`BRAVE_SEARCH_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and an identifying
`CORPUS_FETCH_CONTACT`. No command loads that file implicitly. Pass it explicitly:

```powershell
python ingest_trusted_corpus.py readiness --env-file ..\..\.env.local
```

Existing process variables override file values. Diagnostics never contain API
keys or response bodies.

## Retrieval behavior

Brave results are discovery-only `search_snippet` material. At most three unique
registry-approved source URLs are fetched. Every hop revalidates HTTPS, the
allow-list, redirects, DNS/public IP, content type, size, and the request
deadline. Trafilatura extracts HTML main text (including useful tables, excluding
comments/navigation), and pypdf accepts only digital-text PDFs; OCR is not used.

Fetched pages are chunked with the corpus settings and ranked with the pinned
MiniLM embedding. `source_excerpt` and `corpus_chunk` material can be assessed;
snippets cannot. Rights-eligible fetched pages are staged by content hash in the
ignored `.corpus/staging` directory and never written directly to Qdrant.

Hybrid mode starts Brave/source fetching and Qdrant together inside the one
configured timeout (10 seconds by default). Empty provider results are
`none_found`; unavailable providers remain explicit structured failures.

## Review-gated lifecycle

All mutation commands are dry-run unless `--apply` is supplied:

```powershell
python ingest_trusted_corpus.py single --url https://www.bls.gov/example --rights-reviewer REVIEWER --rights-reviewed-at 2026-07-12
python ingest_trusted_corpus.py manifest-validate corpus_manifests\candidate.json
python ingest_trusted_corpus.py manifest-approve corpus_manifests\candidate.json --reviewer REVIEWER --apply
python ingest_trusted_corpus.py batch --env-file ..\..\.env.local
python ingest_trusted_corpus.py batch --env-file ..\..\.env.local --apply
python ingest_trusted_corpus.py stats --env-file ..\..\.env.local
python ingest_trusted_corpus.py refresh --env-file ..\..\.env.local --apply
python ingest_trusted_corpus.py expiry --env-file ..\..\.env.local
python ingest_trusted_corpus.py expiry --env-file ..\..\.env.local --apply
python ingest_trusted_corpus.py backup --env-file ..\..\.env.local --apply
python ingest_trusted_corpus.py restore-verify --env-file ..\..\.env.local --apply
```

The checked-in 40-document asset is intentionally an `initial-candidate-*`
manifest. Its sources and required families are enumerated, but its content
hashes and human rights reviews remain pending. It cannot become the current
manifest or be ingested until source versions are frozen, staged, and reviewed.

## Scheduled maintenance and live checks

User-level Task Scheduler commands install every-five-day health, weekly
refresh/backup/expiry-dry-run, and monthly isolated restore verification:

```powershell
python ingest_trusted_corpus.py scheduler status
python ingest_trusted_corpus.py scheduler install --env-file ..\..\.env.local --apply
python ingest_trusted_corpus.py scheduler remove --apply
```

Live tests are gated by `--apply`; they do not intentionally burst or exhaust
Brave. Qdrant checks use and clean up a unique temporary collection:

```powershell
python ingest_trusted_corpus.py live-test brave --env-file ..\..\.env.local --apply
python ingest_trusted_corpus.py live-test qdrant --env-file ..\..\.env.local --apply
python ingest_trusted_corpus.py live-test hybrid --env-file ..\..\.env.local --apply
```

## Retrieval-only evaluation

The `retrieval_eval/v1` bootstrap has the required 120 atomic cases and 24
multi-claim scenarios with the required source and split strata. It is marked
pending human review, and the runner refuses held-out evaluation until every
case, pooled top-10 judgment, chunk span, and scenario is reviewed:

```powershell
python retrieval_eval_runner.py --validate-only
python retrieval_eval_runner.py --mode all --env-file ..\..\.env.local --output .corpus\retrieval-report.json
```

The runner evaluates dense retrieval without a threshold first, may select a
development-only threshold, and only then conditionally uses pinned
`cross-encoder/ms-marco-MiniLM-L6-v2` at revision
`c5ee24cb16019beea0893ab7796b1df96625c6b8`. Held-out gates are never weakened:
recall@5 >= 0.80, irrelevant-result rate@5 <= 0.30, multi-claim coverage@10 >=
0.80, and warm hybrid p95 <= 10 seconds.
