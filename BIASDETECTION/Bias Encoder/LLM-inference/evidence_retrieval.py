#!/usr/bin/env python
"""Retrieve strictly trusted web evidence for bias-report context windows.

The public ``retrieve_evidence`` function is retained for existing inference
callers. New callers should construct an ``EvidenceRequest`` and depend on the
``EvidenceRetriever`` protocol instead.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Protocol

import retrieval_store
import source_fetch
import trusted_source_registry


BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_MAX_QUERIES = 3
DEFAULT_POLICY_VERSION = "trusted_sources/v2"
RRF_RANK_CONSTANT = 60
CORPUS_CANDIDATE_MULTIPLIER = 3
SIGNIFICANT_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
STOP_WORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "among",
    "because",
    "before",
    "being",
    "between",
    "could",
    "during",
    "every",
    "from",
    "have",
    "into",
    "more",
    "only",
    "other",
    "over",
    "said",
    "should",
    "some",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "through",
    "under",
    "were",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


@dataclass(frozen=True)
class TrustedSourcePolicy:
    official_suffixes: tuple[str, ...]
    primary_official_domains: tuple[str, ...]
    empirical_research_domains: tuple[str, ...]
    wire_service_domains: tuple[str, ...]
    version: str = DEFAULT_POLICY_VERSION
    source_registry: trusted_source_registry.TrustedSourceRegistry | None = None


@dataclass(frozen=True)
class EvidenceClaim:
    """A report-relevant article claim used to build trusted-web queries."""

    claim_id: str
    text: str
    context_text: str
    selected_sentence_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", self.claim_id.strip())
        object.__setattr__(self, "text", clean_text(self.text))
        object.__setattr__(self, "context_text", clean_text(self.context_text))
        object.__setattr__(
            self,
            "selected_sentence_indices",
            tuple(self.selected_sentence_indices),
        )


@dataclass(frozen=True)
class EvidenceQuery:
    text: str
    source: str
    claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceProvenance:
    """The stable provenance contract shared with future backend consumers."""

    provider: str
    query: str
    query_source: str
    collection: str | None
    document_id: str | None
    chunk_id: str | None
    point_id: str | None
    score: float | None
    extraction_version: str | None = None
    archive_hash: str | None = None


@dataclass(frozen=True)
class EvidenceFailure:
    """Safe, structured retrieval failure metadata."""

    provider: str | None
    stage: str
    message: str
    query: str | None = None
    query_source: str | None = None
    exception_type: str | None = None
    code: str | None = None
    http_status: int | None = None
    retry_after_seconds: float | None = None


@dataclass(frozen=True)
class EvidenceItem:
    title: str
    url: str
    domain: str
    source_type: str
    published_date: str | None
    snippet: str
    query: str
    retrieval_reason: str
    claim_ids: tuple[str, ...] = ()
    canonical_url: str | None = None
    retrieved_at: str | None = None
    provenance: EvidenceProvenance | None = None
    content_hash: str | None = None
    material_kind: str = "search_snippet"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRequest:
    """Stable retriever input for an article or report context window."""

    mode: str
    claims: tuple[EvidenceClaim, ...]
    max_items: int
    timeout_seconds: float
    request_id: str | None = None
    article_id: str | None = None
    policy_version: str | None = None


@dataclass(frozen=True)
class EvidenceResult:
    status: str
    items: list[EvidenceItem]
    error: str | None = None
    requested_mode: str | None = None
    effective_mode: str | None = None
    elapsed_ms: int | None = None
    request_id: str | None = None
    article_id: str | None = None
    policy_version: str | None = None
    provider: str | None = None
    queries: list[EvidenceQuery] = field(default_factory=list)
    failures: list[EvidenceFailure] = field(default_factory=list)
    provider_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def items_json(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index, item in enumerate(self.items, start=1):
            record = item.to_json()
            record["citation_id"] = f"E{index}"
            records.append(record)
        return records

    def metadata_json(self) -> dict[str, Any]:
        """Return additive result metadata suitable for JSONL diagnostics."""

        return {
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "elapsed_ms": self.elapsed_ms,
            "request_id": self.request_id,
            "article_id": self.article_id,
            "policy_version": self.policy_version,
            "provider": self.provider,
            "queries": [asdict(query) for query in self.queries],
            "failures": [asdict(failure) for failure in self.failures],
            "provider_diagnostics": list(self.provider_diagnostics),
        }


class RetrieverUnavailableError(RuntimeError):
    """Raised when a requested retriever mode or provider cannot be selected."""


class BraveTransportError(RuntimeError):
    """Stable Brave failure that never includes response bodies or credentials."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        http_status: int | None = None,
        retry_after_seconds: float | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.retry_after_seconds = retry_after_seconds
        self.diagnostics = diagnostics or {}


class BraveSearchItems(list[EvidenceItem]):
    """Backward-compatible list carrying redacted response diagnostics."""

    def __init__(self, values: list[EvidenceItem], diagnostics: dict[str, Any]) -> None:
        super().__init__(values)
        self.diagnostics = diagnostics


class EvidenceRetriever(Protocol):
    """Stable retrieval interface for inference and future backend consumers."""

    def retrieve(self, request: EvidenceRequest) -> EvidenceResult:
        """Retrieve trusted evidence for report-relevant claims."""


DEFAULT_TRUSTED_SOURCE_POLICY = TrustedSourcePolicy(
    official_suffixes=(
        ".gov",
        ".mil",
    ),
    primary_official_domains=(
        "bea.gov",
        "bls.gov",
        "cdc.gov",
        "census.gov",
        "congress.gov",
        "courts.ca.gov",
        "data.gov",
        "ec.europa.eu",
        "ed.gov",
        "epa.gov",
        "europa.eu",
        "fbi.gov",
        "fec.gov",
        "fda.gov",
        "federalreserve.gov",
        "ftc.gov",
        "gao.gov",
        "house.gov",
        "imf.org",
        "irs.gov",
        "justice.gov",
        "nih.gov",
        "noaa.gov",
        "nsa.gov",
        "oecd.org",
        "sec.gov",
        "senate.gov",
        "treasury.gov",
        "un.org",
        "who.int",
        "worldbank.org",
    ),
    empirical_research_domains=(
        "academic.oup.com",
        "acpjournals.org",
        "annualreviews.org",
        "arxiv.org",
        "bmj.com",
        "cambridge.org",
        "cell.com",
        "cochranelibrary.com",
        "doi.org",
        "elsevier.com",
        "jamanetwork.com",
        "journals.plos.org",
        "link.springer.com",
        "nature.com",
        "nber.org",
        "nejm.org",
        "ncbi.nlm.nih.gov",
        "oup.com",
        "pnas.org",
        "pubmed.ncbi.nlm.nih.gov",
        "science.org",
        "sciencedirect.com",
        "springer.com",
        "ssrn.com",
        "thelancet.com",
        "wiley.com",
    ),
    wire_service_domains=(
        "apnews.com",
        "reuters.com",
    ),
)


def load_trusted_source_policy(path: Path | str | None) -> TrustedSourcePolicy:
    source_path = Path(path) if path is not None else Path(__file__).with_name("trusted_sources.json")
    data = json.loads(source_path.read_text(encoding="utf-8"))
    registry = trusted_source_registry.TrustedSourceRegistry.from_file(source_path)
    return TrustedSourcePolicy(
        official_suffixes=tuple(data.get("official_suffixes", ())),
        primary_official_domains=tuple(data.get("primary_official_domains", ())),
        empirical_research_domains=tuple(data.get("empirical_research_domains", ())),
        wire_service_domains=tuple(
            data.get("wire_service_domains", data.get("wire_factual_domains", ()))
        ),
        version=str(data.get("version") or DEFAULT_POLICY_VERSION),
        source_registry=registry,
    )


def normalize_domain(url_or_domain: str) -> str:
    value = url_or_domain.strip().casefold()
    parsed = urllib.parse.urlparse(value if "://" in value else f"https://{value}")
    domain = parsed.netloc or parsed.path
    if "@" in domain:
        domain = domain.rsplit("@", 1)[-1]
    if ":" in domain:
        domain = domain.split(":", 1)[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.strip(".")


def domain_matches(domain: str, trusted_domain: str) -> bool:
    domain = normalize_domain(domain)
    trusted_domain = normalize_domain(trusted_domain)
    return domain == trusted_domain or domain.endswith(f".{trusted_domain}")


def classify_trusted_source(
    url: str, policy: TrustedSourcePolicy = DEFAULT_TRUSTED_SOURCE_POLICY
) -> str | None:
    domain = normalize_domain(url)
    if any(domain_matches(domain, item) for item in policy.empirical_research_domains):
        return "empirical_research"
    if any(domain_matches(domain, item) for item in policy.wire_service_domains):
        return "wire_service"
    if any(domain.endswith(suffix) for suffix in policy.official_suffixes):
        return "primary_official"
    if any(domain_matches(domain, item) for item in policy.primary_official_domains):
        return "primary_official"
    return None


def canonicalize_web_url(url: str) -> str | None:
    """Normalize a permitted HTTPS URL for in-request deduplication.

    This deliberately does not claim an origin-provided canonical document URL:
    the web-only phase does not fetch documents or follow canonical-link tags.
    """

    try:
        parsed = urllib.parse.urlsplit(url.strip())
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    hostname = parsed.hostname.casefold().rstrip(".")
    if not hostname:
        return None
    netloc = hostname
    if port and port != 443:
        netloc = f"{netloc}:{port}"
    path = parsed.path or "/"
    return urllib.parse.urlunsplit(("https", netloc, path, parsed.query, ""))


def is_permitted_web_url(
    url: str, policy: TrustedSourcePolicy = DEFAULT_TRUSTED_SOURCE_POLICY
) -> bool:
    return canonicalize_web_url(url) is not None and classify_trusted_source(url, policy) is not None


def build_evidence_queries(
    selected_sentence_texts: list[str],
    context_text: str,
    max_queries: int = DEFAULT_MAX_QUERIES,
) -> list[EvidenceQuery]:
    """Compatibility query builder for callers that do not yet pass claims."""

    claims = tuple(
        EvidenceClaim(
            claim_id=f"claim-{index}",
            text=sentence,
            context_text=context_text,
        )
        for index, sentence in enumerate(selected_sentence_texts, start=1)
    )
    if not claims and context_text:
        claims = (
            EvidenceClaim(
                claim_id="context-1",
                text="",
                context_text=context_text,
            ),
        )
    return build_evidence_queries_for_claims(claims, max_queries=max_queries)


def build_evidence_queries_for_claims(
    claims: tuple[EvidenceClaim, ...],
    max_queries: int = DEFAULT_MAX_QUERIES,
) -> list[EvidenceQuery]:
    queries: list[EvidenceQuery] = []
    if max_queries <= 0:
        return queries

    for claim in claims:
        query_text = compact_query(claim.text)
        if query_text:
            queries.append(
                EvidenceQuery(
                    text=query_text,
                    source="selected_sentence",
                    claim_ids=(claim.claim_id,),
                )
            )
        if len(queries) >= max_queries:
            return dedupe_queries(queries)[:max_queries]

    for claim in claims:
        context_query = compact_query(claim.context_text)
        if context_query:
            queries.append(
                EvidenceQuery(
                    text=context_query,
                    source="context_window",
                    claim_ids=(claim.claim_id,),
                )
            )
        if len(queries) >= max_queries:
            break
    return dedupe_queries(queries)[:max_queries]


def compact_query(text: str, max_terms: int = 14) -> str:
    words = [
        match.group(0)
        for match in SIGNIFICANT_WORD_RE.finditer(text.replace("â€™", "'"))
        if match.group(0).casefold() not in STOP_WORDS
    ]
    query_words: list[str] = []
    seen: set[str] = set()
    for word in words:
        key = word.casefold()
        if key in seen:
            continue
        seen.add(key)
        query_words.append(word)
        if len(query_words) >= max_terms:
            break
    return " ".join(query_words)


def dedupe_queries(queries: list[EvidenceQuery]) -> list[EvidenceQuery]:
    deduped: list[EvidenceQuery] = []
    seen: dict[str, int] = {}
    for query in queries:
        key = query.text.casefold()
        existing_index = seen.get(key)
        if existing_index is None:
            seen[key] = len(deduped)
            deduped.append(query)
            continue
        existing = deduped[existing_index]
        deduped[existing_index] = replace(
            existing,
            claim_ids=_merge_ids(existing.claim_ids, query.claim_ids),
        )
    return deduped


def select_evidence_retriever(
    mode: str,
    *,
    provider: str = "brave",
    policy: TrustedSourcePolicy = DEFAULT_TRUSTED_SOURCE_POLICY,
    api_key: str | None = None,
    corpus_config: retrieval_store.QdrantCorpusConfig | None = None,
    corpus_store: retrieval_store.QdrantCorpusStore | None = None,
    corpus_embedding_provider: retrieval_store.EmbeddingProvider | None = None,
) -> EvidenceRetriever:
    """Select the configured retriever without exposing provider details to callers."""

    normalized_mode = str(mode).casefold()
    if normalized_mode == "off":
        return DisabledEvidenceRetriever(policy=policy)
    if normalized_mode == "web":
        if provider.casefold() != "brave":
            raise RetrieverUnavailableError(f"Unsupported evidence provider: {provider}")
        return BraveWebEvidenceRetriever(policy=policy, api_key=api_key)
    if normalized_mode == "corpus":
        return CorpusEvidenceRetriever(
            policy=policy,
            config=corpus_config,
            store=corpus_store,
            embedding_provider=corpus_embedding_provider,
        )
    if normalized_mode == "hybrid":
        if provider.casefold() != "brave":
            raise RetrieverUnavailableError(f"Unsupported evidence provider: {provider}")
        return HybridEvidenceRetriever(
            policy=policy,
            web_retriever=BraveWebEvidenceRetriever(policy=policy, api_key=api_key),
            corpus_retriever=CorpusEvidenceRetriever(
                policy=policy,
                config=corpus_config,
                store=corpus_store,
                embedding_provider=corpus_embedding_provider,
            ),
        )
    raise RetrieverUnavailableError(f"Unsupported evidence mode: {mode}")


class DisabledEvidenceRetriever:
    """A no-op retriever for explicit off mode."""

    def __init__(self, policy: TrustedSourcePolicy) -> None:
        self.policy = policy

    def retrieve(self, request: EvidenceRequest) -> EvidenceResult:
        started = time.monotonic()
        return EvidenceResult(
            status="not_requested",
            items=[],
            requested_mode=request.mode,
            effective_mode="off",
            elapsed_ms=_elapsed_ms(started),
            request_id=request.request_id,
            article_id=request.article_id,
            policy_version=request.policy_version or self.policy.version,
        )


class BraveWebEvidenceRetriever:
    """Brave discovery retriever with strict trusted-source enforcement."""

    provider = "brave"

    def __init__(
        self,
        *,
        policy: TrustedSourcePolicy = DEFAULT_TRUSTED_SOURCE_POLICY,
        api_key: str | None = None,
        fetch_contact: str | None = None,
        source_fetcher: Any | None = None,
        source_embedding_provider: retrieval_store.EmbeddingProvider | None = None,
        staging_dir: Path | str | None = None,
    ) -> None:
        self.policy = policy
        self.api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY")
        self.fetch_contact = fetch_contact or os.environ.get("CORPUS_FETCH_CONTACT")
        self.source_fetcher = source_fetcher or source_fetch.fetch_source_document
        self._source_embedding_provider = source_embedding_provider
        self.staging_dir = Path(staging_dir) if staging_dir is not None else (
            Path(__file__).with_name(".corpus") / "staging"
        )

    def retrieve(self, request: EvidenceRequest) -> EvidenceResult:
        started = time.monotonic()
        deadline = started + max(0.0, float(request.timeout_seconds))
        metadata = _request_metadata(request, self.policy, self.provider)
        if request.mode.casefold() != "web":
            return _error_result(
                metadata=metadata,
                started=started,
                error="Brave web retriever requires evidence mode 'web'",
                failure=EvidenceFailure(
                    provider=self.provider,
                    stage="selection",
                    message="Brave web retriever requires evidence mode 'web'",
                    code="invalid_mode",
                ),
            )
        if request.max_items <= 0:
            return _result(
                status="none_found",
                items=[],
                metadata=metadata,
                started=started,
            )
        if request.timeout_seconds <= 0:
            return _error_result(
                metadata=metadata,
                started=started,
                error="Evidence timeout_seconds must be greater than zero",
                failure=EvidenceFailure(
                    provider=self.provider,
                    stage="validation",
                    message="Evidence timeout_seconds must be greater than zero",
                    code="invalid_request",
                ),
            )
        queries = build_evidence_queries_for_claims(request.claims)
        metadata["queries"] = queries
        if not queries:
            return _result(
                status="none_found",
                items=[],
                metadata=metadata,
                started=started,
            )
        if not self.api_key:
            return _error_result(
                metadata=metadata,
                started=started,
                error="BRAVE_SEARCH_API_KEY is not set",
                failure=EvidenceFailure(
                    provider=self.provider,
                    stage="configuration",
                    message="BRAVE_SEARCH_API_KEY is not set",
                    exception_type="RetrieverUnavailableError",
                    code="configuration",
                ),
            )

        items: list[EvidenceItem] = []
        seen_urls: dict[str, int] = {}
        failures: list[EvidenceFailure] = []
        diagnostics: list[dict[str, Any]] = []
        discovery_limit = max(request.max_items, 3)
        for query in queries:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BraveTransportError("timeout", "Brave search deadline expired")
                candidates = search_brave(
                    query=query,
                    api_key=self.api_key,
                    timeout_seconds=remaining,
                    policy=self.policy,
                )
                response_diagnostics = getattr(candidates, "diagnostics", None)
                if isinstance(response_diagnostics, dict):
                    diagnostics.append(response_diagnostics)
            except Exception as exc:
                failure = _provider_failure(
                    exc,
                    provider=self.provider,
                    stage="search",
                    message="Brave web search failed",
                    query=query.text,
                    query_source=query.source,
                )
                failures.append(failure)
                if isinstance(exc, BraveTransportError) and exc.diagnostics:
                    diagnostics.append(exc.diagnostics)
                break

            for candidate in candidates:
                accepted = self._accept_candidate(candidate, query)
                if accepted is None:
                    continue
                canonical_url = accepted.canonical_url
                if canonical_url is None:
                    continue
                existing_index = seen_urls.get(canonical_url)
                if existing_index is not None:
                    previous = items[existing_index]
                    items[existing_index] = replace(
                        previous,
                        claim_ids=_merge_ids(previous.claim_ids, accepted.claim_ids),
                    )
                    continue
                seen_urls[canonical_url] = len(items)
                items.append(accepted)
                if len(items) >= discovery_limit:
                    break
            if len(items) >= discovery_limit:
                break

        excerpts: list[EvidenceItem] = []
        if items and self.policy.source_registry is not None:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="source-fetch")
            future = executor.submit(
                self._fetch_source_excerpts,
                items,
                request,
                deadline=deadline,
            )
            try:
                remaining = max(0.0, deadline - time.monotonic())
                fetched_items, fetch_failures, fetch_diagnostics = future.result(timeout=remaining)
                excerpts.extend(fetched_items)
                failures.extend(fetch_failures)
                diagnostics.extend(fetch_diagnostics)
            except FutureTimeoutError:
                future.cancel()
                failures.append(EvidenceFailure(
                    provider="source_fetch",
                    stage="fetch",
                    message="Trusted source extraction exceeded the evidence deadline",
                    code="timeout",
                ))
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        ranked_items = _deduplicate_evidence_items(excerpts + items)[: request.max_items]
        error_message = None
        if failures:
            error_message = (
                "Brave web search failed"
                if any(failure.provider == "brave" and failure.stage == "search" for failure in failures)
                else "One or more web evidence stages failed"
            )
        return _result(
            status=("partial" if failures and ranked_items else "error" if failures else "found" if ranked_items else "none_found"),
            items=ranked_items,
            error=error_message,
            metadata=metadata,
            started=started,
            failures=failures,
            provider_diagnostics=diagnostics,
        )

    def _fetch_source_excerpts(
        self,
        snippets: list[EvidenceItem],
        request: EvidenceRequest,
        *,
        deadline: float,
    ) -> tuple[list[EvidenceItem], list[EvidenceFailure], list[dict[str, Any]]]:
        registry = self.policy.source_registry
        if registry is None:
            return [], [], []
        if not str(self.fetch_contact or "").strip():
            return [], [
                EvidenceFailure(
                    provider="source_fetch",
                    stage="configuration",
                    message="CORPUS_FETCH_CONTACT is not set",
                    code="configuration",
                )
            ], []
        unique: list[EvidenceItem] = []
        seen: set[str] = set()
        for item in snippets:
            canonical = item.canonical_url or item.url
            decision = registry.evaluate_url(canonical, purpose="fetch")
            if not decision.allowed or not decision.canonical_url or decision.canonical_url in seen:
                continue
            seen.add(decision.canonical_url)
            unique.append(item)
            if len(unique) == 3:
                break

        excerpts: list[EvidenceItem] = []
        failures: list[EvidenceFailure] = []
        diagnostics: list[dict[str, Any]] = []
        for discovery in unique:
            try:
                if deadline <= time.monotonic():
                    raise source_fetch.SourceFetchError("timeout", "Source fetch deadline expired")
                document = self.source_fetcher(
                    discovery.canonical_url or discovery.url,
                    registry,
                    contact=str(self.fetch_contact),
                    deadline=deadline,
                )
                decision = registry.evaluate_url(document.canonical_url, purpose="fetch")
                if not decision.allowed or decision.rule is None:
                    raise source_fetch.SourceFetchError("policy_rejected", "Extracted canonical URL was rejected")
                if decision.rule.ingest_allowed and decision.rule.redistribution_allowed:
                    staged = source_fetch.stage_source_document(
                        document,
                        decision.rule,
                        staging_dir=self.staging_dir,
                    )
                    diagnostics.append({
                        "provider": "source_fetch",
                        "stage_id": staged.stage_id,
                        "rights_status": staged.rights_status,
                    })
                excerpts.extend(self._rank_document_chunks(document, discovery, request.claims))
            except Exception as exc:
                failures.append(
                    _provider_failure(
                        exc,
                        provider="source_fetch",
                        stage="fetch",
                        message="Trusted source extraction failed",
                        query=discovery.query,
                        query_source=discovery.retrieval_reason,
                    )
                )
        return excerpts, failures, diagnostics

    def _rank_document_chunks(
        self,
        document: source_fetch.ExtractedSourceDocument,
        discovery: EvidenceItem,
        claims: tuple[EvidenceClaim, ...],
    ) -> list[EvidenceItem]:
        chunks = retrieval_store.chunk_document_text(document.text)
        relevant_claims = [
            claim for claim in claims
            if not discovery.claim_ids or claim.claim_id in discovery.claim_ids
        ] or list(claims)
        if not relevant_claims:
            return []
        embedder = self._resolve_source_embedding_provider()
        texts = [claim.text or claim.context_text for claim in relevant_claims] + [
            chunk.text for chunk in chunks
        ]
        vectors = embedder.embed(texts)
        if len(vectors) != len(texts):
            raise ValueError("Source excerpt embedder returned an unexpected vector count")
        claim_vectors = vectors[: len(relevant_claims)]
        chunk_vectors = vectors[len(relevant_claims):]
        document_id = retrieval_store.stable_document_id(document.canonical_url)
        results: list[EvidenceItem] = []
        for claim, claim_vector in zip(relevant_claims, claim_vectors, strict=True):
            scored = [
                (_cosine_similarity(claim_vector, vector), chunk)
                for chunk, vector in zip(chunks, chunk_vectors, strict=True)
            ]
            score, chunk = max(scored, key=lambda value: (value[0], -value[1].index))
            chunk_hash = retrieval_store.sha256_text(chunk.text)
            chunk_id = retrieval_store.stable_chunk_id(document_id, chunk.index, chunk_hash)
            results.append(
                EvidenceItem(
                    title=document.title,
                    url=document.canonical_url,
                    domain=normalize_domain(document.canonical_url),
                    source_type=discovery.source_type,
                    published_date=document.published_at or discovery.published_date,
                    snippet=chunk.text,
                    query=claim.text or claim.context_text,
                    retrieval_reason="source_page_chunk",
                    claim_ids=(claim.claim_id,),
                    canonical_url=document.canonical_url,
                    retrieved_at=document.fetched_at,
                    provenance=EvidenceProvenance(
                        provider="source_fetch",
                        query=claim.text or claim.context_text,
                        query_source="source_page_chunk",
                        collection=None,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        point_id=None,
                        score=score,
                        extraction_version=document.extraction_version,
                        archive_hash=document.archive_hash,
                    ),
                    content_hash=document.content_hash,
                    material_kind="source_excerpt",
                )
            )
        return results

    def _resolve_source_embedding_provider(self) -> retrieval_store.EmbeddingProvider:
        if self._source_embedding_provider is not None:
            return self._source_embedding_provider
        model_name = retrieval_store.DEFAULT_EMBEDDING_MODEL
        cached = _CORPUS_EMBEDDER_CACHE.get(model_name)
        if cached is None:
            from ingest_trusted_corpus import SentenceTransformerEmbedder

            cached = SentenceTransformerEmbedder(model_name)
            _CORPUS_EMBEDDER_CACHE[model_name] = cached
        self._source_embedding_provider = cached
        return cached

    def _accept_candidate(
        self,
        candidate: EvidenceItem,
        query: EvidenceQuery,
    ) -> EvidenceItem | None:
        """Re-enforce the policy even when a provider helper is mocked or replaced."""

        if not is_permitted_web_url(candidate.url, self.policy):
            return None
        source_type = classify_trusted_source(candidate.url, self.policy)
        canonical_url = canonicalize_web_url(candidate.url)
        if source_type is None or canonical_url is None:
            return None
        return replace(
            candidate,
            domain=normalize_domain(candidate.url),
            source_type=source_type,
            query=query.text,
            retrieval_reason=query.source,
            claim_ids=_merge_ids(candidate.claim_ids, query.claim_ids),
            canonical_url=canonical_url,
            retrieved_at=candidate.retrieved_at or _utc_now(),
            provenance=EvidenceProvenance(
                provider=self.provider,
                query=query.text,
                query_source=query.source,
                collection=None,
                document_id=None,
                chunk_id=None,
                point_id=None,
                score=None,
            ),
        )


_CORPUS_STORE_CACHE: dict[
    retrieval_store.QdrantCorpusConfig, retrieval_store.QdrantCorpusStore
] = {}
_CORPUS_EMBEDDER_CACHE: dict[str, retrieval_store.EmbeddingProvider] = {}


class CorpusEvidenceRetriever:
    """Dense retrieval over the optional trusted Qdrant corpus."""

    provider = "qdrant"

    def __init__(
        self,
        *,
        policy: TrustedSourcePolicy = DEFAULT_TRUSTED_SOURCE_POLICY,
        config: retrieval_store.QdrantCorpusConfig | None = None,
        store: retrieval_store.QdrantCorpusStore | None = None,
        embedding_provider: retrieval_store.EmbeddingProvider | None = None,
    ) -> None:
        self.policy = policy
        self._config = config
        self._store = store
        self._embedding_provider = embedding_provider

    def retrieve(self, request: EvidenceRequest) -> EvidenceResult:
        started = time.monotonic()
        metadata = _request_metadata(request, self.policy, self.provider)
        metadata["effective_mode"] = "corpus"
        if request.mode.casefold() != "corpus":
            return _error_result(
                metadata=metadata,
                started=started,
                error="Corpus retriever requires evidence mode 'corpus'",
                failure=EvidenceFailure(
                    provider=self.provider,
                    stage="selection",
                    message="Corpus retriever requires evidence mode 'corpus'",
                    code="invalid_mode",
                ),
                effective_mode="off",
            )
        if request.max_items <= 0:
            return _result(
                status="none_found",
                items=[],
                metadata=metadata,
                started=started,
            )
        if request.timeout_seconds <= 0:
            return _error_result(
                metadata=metadata,
                started=started,
                error="Evidence timeout_seconds must be greater than zero",
                failure=EvidenceFailure(
                    provider=self.provider,
                    stage="validation",
                    message="Evidence timeout_seconds must be greater than zero",
                    code="invalid_request",
                ),
                effective_mode="off",
            )

        queries = build_evidence_queries_for_claims(request.claims)
        metadata["queries"] = queries
        if not queries:
            return _result(
                status="none_found",
                items=[],
                metadata=metadata,
                started=started,
            )

        try:
            store = self._resolve_store()
            embedding_provider = self._resolve_embedding_provider(store.config)
            vectors = embedding_provider.embed([query.text for query in queries])
            if len(vectors) != len(queries):
                raise ValueError("Corpus embedding provider returned an unexpected vector count")
        except Exception as exc:
            return _error_result(
                metadata=metadata,
                started=started,
                error="Corpus retrieval is unavailable",
                failure=EvidenceFailure(
                    provider=self.provider,
                    stage="initialization",
                    message="Corpus retrieval could not be initialized",
                    exception_type=type(exc).__name__,
                    code="unavailable",
                ),
                effective_mode="off",
            )

        candidates: list[tuple[EvidenceItem, int, float]] = []
        failures: list[EvidenceFailure] = []
        candidate_limit = max(request.max_items, request.max_items * CORPUS_CANDIDATE_MULTIPLIER)
        for query_index, (query, vector) in enumerate(zip(queries, vectors, strict=True)):
            try:
                points = store.search_points(vector, limit=candidate_limit)
            except Exception as exc:
                failures.append(
                    EvidenceFailure(
                        provider=self.provider,
                        stage="search",
                        message="Qdrant corpus search failed",
                        query=query.text,
                        query_source=query.source,
                        exception_type=type(exc).__name__,
                        code="provider_failure",
                    )
                )
                break
            for point in points:
                item = _corpus_evidence_item(
                    point,
                    query,
                    collection=store.config.collection,
                    content_field=store.config.content_field,
                )
                if item is not None:
                    candidates.append((item, query_index, point.score))

        ranked_items = [
            item
            for item, _query_index, _score in sorted(
                candidates,
                key=lambda value: (-value[2], value[1], _stable_item_identity(value[0])),
            )
        ]
        items = _deduplicate_evidence_items(ranked_items)[: request.max_items]
        if failures:
            return _result(
                status="partial" if items else "error",
                items=items,
                error="Qdrant corpus search failed",
                metadata=metadata,
                started=started,
                failures=failures,
            )
        return _result(
            status="found" if items else "none_found",
            items=items,
            metadata=metadata,
            started=started,
        )

    def _resolve_store(self) -> retrieval_store.QdrantCorpusStore:
        if self._store is not None:
            return self._store
        config = self._config or retrieval_store.QdrantCorpusConfig.from_environment()
        if config.mode == "disabled":
            raise retrieval_store.QdrantUnavailableError("Qdrant is not configured")
        store = _CORPUS_STORE_CACHE.get(config)
        if store is None:
            store = retrieval_store.QdrantCorpusStore(config)
            _CORPUS_STORE_CACHE[config] = store
        return store

    def _resolve_embedding_provider(
        self,
        config: retrieval_store.QdrantCorpusConfig,
    ) -> retrieval_store.EmbeddingProvider:
        if self._embedding_provider is not None:
            return self._embedding_provider
        cached = _CORPUS_EMBEDDER_CACHE.get(config.embedding_model)
        if cached is not None:
            return cached
        try:
            from ingest_trusted_corpus import SentenceTransformerEmbedder

            embedding_provider = SentenceTransformerEmbedder(config.embedding_model)
        except Exception as exc:
            raise RuntimeError("Could not initialize corpus embeddings") from exc
        _CORPUS_EMBEDDER_CACHE[config.embedding_model] = embedding_provider
        return embedding_provider


class HybridEvidenceRetriever:
    """Fuse independent trusted-web and corpus result lists safely."""

    provider = "hybrid"

    def __init__(
        self,
        *,
        policy: TrustedSourcePolicy,
        web_retriever: EvidenceRetriever,
        corpus_retriever: EvidenceRetriever,
    ) -> None:
        self.policy = policy
        self.web_retriever = web_retriever
        self.corpus_retriever = corpus_retriever

    def retrieve(self, request: EvidenceRequest) -> EvidenceResult:
        started = time.monotonic()
        metadata = _request_metadata(request, self.policy, self.provider)
        metadata["effective_mode"] = "hybrid"
        if request.mode.casefold() != "hybrid":
            return _error_result(
                metadata=metadata,
                started=started,
                error="Hybrid retriever requires evidence mode 'hybrid'",
                failure=EvidenceFailure(
                    provider=self.provider,
                    stage="selection",
                    message="Hybrid retriever requires evidence mode 'hybrid'",
                    code="invalid_mode",
                ),
                effective_mode="off",
            )
        if request.max_items <= 0:
            return _result(
                status="none_found",
                items=[],
                metadata=metadata,
                started=started,
            )
        if request.timeout_seconds <= 0:
            return _error_result(
                metadata=metadata,
                started=started,
                error="Evidence timeout_seconds must be greater than zero",
                failure=EvidenceFailure(
                    provider=self.provider,
                    stage="validation",
                    message="Evidence timeout_seconds must be greater than zero",
                    code="invalid_request",
                ),
                effective_mode="off",
            )

        deadline = started + request.timeout_seconds
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="evidence-provider")
        futures: dict[str, Future[EvidenceResult]] = {
            "brave": executor.submit(
                _safe_provider_call,
                self.web_retriever,
                replace(request, mode="web"),
                fallback_provider="brave",
            ),
            "qdrant": executor.submit(
                _safe_provider_call,
                self.corpus_retriever,
                replace(request, mode="corpus"),
                fallback_provider="qdrant",
            ),
        }
        collected: list[tuple[str, EvidenceResult]] = []
        try:
            for provider_name in ("brave", "qdrant"):
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    result = futures[provider_name].result(timeout=remaining)
                except FutureTimeoutError:
                    futures[provider_name].cancel()
                    result = EvidenceResult(
                        status="error",
                        items=[],
                        error="Evidence provider exceeded the hybrid deadline",
                        requested_mode=provider_name,
                        effective_mode="off",
                        provider=provider_name,
                        failures=[EvidenceFailure(
                            provider=provider_name,
                            stage="dispatch",
                            message="Evidence provider exceeded the hybrid deadline",
                            code="timeout",
                        )],
                    )
                collected.append((provider_name, result))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        provider_results = tuple(collected)
        metadata["queries"] = dedupe_queries(
            [query for _provider, result in provider_results for query in result.queries]
        )
        failures: list[EvidenceFailure] = []
        provider_diagnostics: list[dict[str, Any]] = []
        for fallback_provider, result in provider_results:
            provider_diagnostics.extend(result.provider_diagnostics)
            if result.failures:
                failures.extend(result.failures)
            elif result.status in {"error", "partial"}:
                failures.append(
                    EvidenceFailure(
                        provider=result.provider or fallback_provider,
                        stage="provider",
                        message="Evidence provider failed",
                        code="provider_failure",
                    )
                )

        items = _fuse_evidence_rankings(
            [result.items for _provider, result in provider_results],
            max_items=request.max_items,
        )
        if items:
            status = "partial" if failures else "found"
        elif failures:
            status = "error"
        else:
            status = "none_found"
        return _result(
            status=status,
            items=items,
            error="One or more evidence providers failed" if failures else None,
            metadata=metadata,
            started=started,
            failures=failures,
            provider_diagnostics=provider_diagnostics,
        )


@dataclass
class _EvidenceIdentityGroup:
    item: EvidenceItem
    keys: set[str]
    rrf_score: float
    best_rank: int
    best_provider_order: int
    representative_rank: int
    representative_provider_order: int
    active: bool = True


def _safe_provider_call(
    retriever: EvidenceRetriever,
    request: EvidenceRequest,
    *,
    fallback_provider: str,
) -> EvidenceResult:
    started = time.monotonic()
    try:
        return retriever.retrieve(request)
    except Exception as exc:
        return EvidenceResult(
            status="error",
            items=[],
            error="Evidence provider failed",
            requested_mode=request.mode,
            effective_mode="off",
            elapsed_ms=_elapsed_ms(started),
            request_id=request.request_id,
            article_id=request.article_id,
            policy_version=request.policy_version,
            provider=fallback_provider,
            failures=[
                EvidenceFailure(
                    provider=fallback_provider,
                    stage="dispatch",
                    message="Evidence provider failed",
                    exception_type=type(exc).__name__,
                    code="provider_failure",
                )
            ],
        )


def _corpus_evidence_item(
    point: retrieval_store.ScoredCorpusPoint,
    query: EvidenceQuery,
    *,
    collection: str,
    content_field: str,
) -> EvidenceItem | None:
    payload = point.payload
    canonical_url = retrieval_store.canonicalize_document_url(
        str(payload.get("canonical_url") or payload.get("source_url") or "")
    )
    if canonical_url is None:
        return None
    snippet = clean_text(str(payload.get(content_field) or ""))
    if not snippet:
        return None
    document_id = clean_text(str(payload.get("document_id") or "")) or None
    chunk_id = clean_text(str(payload.get("chunk_id") or "")) or None
    point_id = clean_text(str(payload.get("point_id") or point.id)) or point.id
    content_hash = clean_text(str(payload.get("content_hash") or "")) or None
    published_date = clean_text(str(payload.get("published_at") or "")) or None
    source_type = clean_text(str(payload.get("source_tier") or "trusted_corpus"))
    return EvidenceItem(
        title=clean_text(str(payload.get("title") or canonical_url)),
        url=canonical_url,
        domain=clean_text(str(payload.get("source_domain") or ""))
        or normalize_domain(canonical_url),
        source_type=source_type or "trusted_corpus",
        published_date=published_date,
        snippet=snippet,
        query=query.text,
        retrieval_reason=query.source,
        claim_ids=query.claim_ids,
        canonical_url=canonical_url,
        retrieved_at=_utc_now(),
        provenance=EvidenceProvenance(
            provider="qdrant",
            query=query.text,
            query_source=query.source,
            collection=collection,
            document_id=document_id,
            chunk_id=chunk_id,
            point_id=point_id,
            score=point.score,
        ),
        content_hash=content_hash,
        material_kind="corpus_chunk",
    )


def _fuse_evidence_rankings(
    provider_rankings: list[list[EvidenceItem]],
    *,
    max_items: int,
) -> list[EvidenceItem]:
    if max_items <= 0:
        return []
    groups: list[_EvidenceIdentityGroup] = []
    key_to_group: dict[str, _EvidenceIdentityGroup] = {}
    for provider_order, ranking in enumerate(provider_rankings):
        for rank, item in enumerate(_deduplicate_evidence_items(ranking), start=1):
            _add_to_identity_groups(
                groups,
                key_to_group,
                item,
                rank=rank,
                provider_order=provider_order,
                rrf_contribution=1.0 / (RRF_RANK_CONSTANT + rank),
            )
    ordered_groups = sorted(
        (group for group in groups if group.active),
        key=lambda group: (
            -group.rrf_score,
            group.best_rank,
            group.best_provider_order,
            _stable_item_identity(group.item),
        ),
    )
    return [group.item for group in ordered_groups[:max_items]]


def _deduplicate_evidence_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    groups: list[_EvidenceIdentityGroup] = []
    key_to_group: dict[str, _EvidenceIdentityGroup] = {}
    for rank, item in enumerate(items, start=1):
        _add_to_identity_groups(
            groups,
            key_to_group,
            item,
            rank=rank,
            provider_order=0,
            rrf_contribution=0.0,
        )
    return [
        group.item
        for group in sorted(
            (group for group in groups if group.active),
            key=lambda group: (
                group.best_rank,
                group.best_provider_order,
                _stable_item_identity(group.item),
            ),
        )
    ]


def _add_to_identity_groups(
    groups: list[_EvidenceIdentityGroup],
    key_to_group: dict[str, _EvidenceIdentityGroup],
    item: EvidenceItem,
    *,
    rank: int,
    provider_order: int,
    rrf_contribution: float,
) -> None:
    keys = _evidence_identity_keys(item)
    matches: list[_EvidenceIdentityGroup] = []
    seen_groups: set[int] = set()
    for key in keys:
        group = key_to_group.get(key)
        if group is not None and group.active and id(group) not in seen_groups:
            seen_groups.add(id(group))
            matches.append(group)

    if not matches:
        group = _EvidenceIdentityGroup(
            item=item,
            keys=set(keys),
            rrf_score=rrf_contribution,
            best_rank=rank,
            best_provider_order=provider_order,
            representative_rank=rank,
            representative_provider_order=provider_order,
        )
        groups.append(group)
    else:
        group = min(
            matches,
            key=lambda value: (
                value.best_rank,
                value.best_provider_order,
                _stable_item_identity(value.item),
            ),
        )
        for other in matches:
            if other is not group:
                _merge_identity_groups(group, other, key_to_group)

        merged_claim_ids = _merge_ids(group.item.claim_ids, item.claim_ids)
        if _candidate_precedes(
            item,
            rank,
            provider_order,
            group.item,
            group.representative_rank,
            group.representative_provider_order,
        ):
            group.item = replace(item, claim_ids=merged_claim_ids)
            group.representative_rank = rank
            group.representative_provider_order = provider_order
        else:
            group.item = replace(group.item, claim_ids=merged_claim_ids)
        group.rrf_score += rrf_contribution
        if (rank, provider_order, _stable_item_identity(item)) < (
            group.best_rank,
            group.best_provider_order,
            _stable_item_identity(group.item),
        ):
            group.best_rank = rank
            group.best_provider_order = provider_order
        group.keys.update(keys)

    for key in group.keys:
        key_to_group[key] = group


def _merge_identity_groups(
    primary: _EvidenceIdentityGroup,
    secondary: _EvidenceIdentityGroup,
    key_to_group: dict[str, _EvidenceIdentityGroup],
) -> None:
    merged_claim_ids = _merge_ids(primary.item.claim_ids, secondary.item.claim_ids)
    if _candidate_precedes(
        secondary.item,
        secondary.representative_rank,
        secondary.representative_provider_order,
        primary.item,
        primary.representative_rank,
        primary.representative_provider_order,
    ):
        primary.item = replace(secondary.item, claim_ids=merged_claim_ids)
        primary.representative_rank = secondary.representative_rank
        primary.representative_provider_order = secondary.representative_provider_order
    else:
        primary.item = replace(primary.item, claim_ids=merged_claim_ids)
    primary.rrf_score += secondary.rrf_score
    if (
        secondary.best_rank,
        secondary.best_provider_order,
        _stable_item_identity(secondary.item),
    ) < (
        primary.best_rank,
        primary.best_provider_order,
        _stable_item_identity(primary.item),
    ):
        primary.best_rank = secondary.best_rank
        primary.best_provider_order = secondary.best_provider_order
    primary.keys.update(secondary.keys)
    for key in secondary.keys:
        key_to_group[key] = primary
    secondary.active = False


def _candidate_precedes(
    item: EvidenceItem,
    rank: int,
    provider_order: int,
    existing_item: EvidenceItem,
    existing_rank: int,
    existing_provider_order: int,
) -> bool:
    return (_material_priority(item), rank, provider_order, _stable_item_identity(item)) < (
        _material_priority(existing_item),
        existing_rank,
        existing_provider_order,
        _stable_item_identity(existing_item),
    )


def _material_priority(item: EvidenceItem) -> int:
    return {
        "source_excerpt": 0,
        "corpus_chunk": 0,
        "search_snippet": 1,
    }.get(str(item.material_kind or "").casefold(), 2)


def _evidence_identity_keys(item: EvidenceItem) -> tuple[str, ...]:
    keys: set[str] = set()
    canonical_url = retrieval_store.canonicalize_document_url(
        item.canonical_url or item.url
    )
    if canonical_url:
        keys.add(f"url:{canonical_url}")
    if item.provenance and item.provenance.document_id:
        keys.add(f"document:{item.provenance.document_id.strip()}")
    if item.content_hash:
        keys.add(f"content:{item.content_hash.strip().casefold()}")
    if not keys:
        keys.add(
            "fallback:"
            + "\\0".join(
                (
                    clean_text(item.url).casefold(),
                    clean_text(item.title).casefold(),
                    clean_text(item.snippet).casefold(),
                )
            )
        )
    return tuple(sorted(keys))


def _stable_item_identity(item: EvidenceItem) -> str:
    return _evidence_identity_keys(item)[0]


def retrieve_evidence_request(
    request: EvidenceRequest,
    *,
    provider: str = "brave",
    policy: TrustedSourcePolicy = DEFAULT_TRUSTED_SOURCE_POLICY,
    api_key: str | None = None,
    corpus_config: retrieval_store.QdrantCorpusConfig | None = None,
    corpus_store: retrieval_store.QdrantCorpusStore | None = None,
    corpus_embedding_provider: retrieval_store.EmbeddingProvider | None = None,
) -> EvidenceResult:
    """Dispatch an EvidenceRequest and convert unavailable providers safely."""

    started = time.monotonic()
    try:
        retriever = select_evidence_retriever(
            request.mode,
            provider=provider,
            policy=policy,
            api_key=api_key,
            corpus_config=corpus_config,
            corpus_store=corpus_store,
            corpus_embedding_provider=corpus_embedding_provider,
        )
        return retriever.retrieve(request)
    except RetrieverUnavailableError as exc:
        message = str(exc)
        return _error_result(
            metadata=_request_metadata(request, policy, provider),
            started=started,
            error=message,
            failure=EvidenceFailure(
                provider=provider,
                stage="selection",
                message=message,
                exception_type=type(exc).__name__,
                code="unavailable",
            ),
            effective_mode="off",
        )
    except Exception as exc:
        return _error_result(
            metadata=_request_metadata(request, policy, provider),
            started=started,
            error="Evidence retrieval failed",
            failure=EvidenceFailure(
                provider=provider,
                stage="dispatch",
                message="Evidence retrieval failed",
                exception_type=type(exc).__name__,
                code="provider_failure",
            ),
            effective_mode="off",
        )


def retrieve_evidence(
    selected_sentence_texts: list[str],
    context_text: str,
    provider: str,
    max_items: int,
    timeout_seconds: float,
    policy: TrustedSourcePolicy = DEFAULT_TRUSTED_SOURCE_POLICY,
    api_key: str | None = None,
    *,
    mode: str = "web",
    request_id: str | None = None,
    article_id: str | None = None,
    claims: tuple[EvidenceClaim, ...] | None = None,
    corpus_config: retrieval_store.QdrantCorpusConfig | None = None,
    corpus_store: retrieval_store.QdrantCorpusStore | None = None,
    corpus_embedding_provider: retrieval_store.EmbeddingProvider | None = None,
) -> EvidenceResult:
    """Compatibility facade for existing Brave callers.

    Existing positional and keyword arguments retain their behavior. New code
    should call ``retrieve_evidence_request`` with explicit ``EvidenceClaim``
    values instead.
    """

    if claims is None:
        claims = tuple(
            EvidenceClaim(
                claim_id=f"claim-{index}",
                text=sentence,
                context_text=context_text,
            )
            for index, sentence in enumerate(selected_sentence_texts, start=1)
        )
        if not claims and context_text:
            claims = (
                EvidenceClaim(
                    claim_id="context-1",
                    text="",
                    context_text=context_text,
                ),
            )
    else:
        claims = tuple(claims)
    request = EvidenceRequest(
        mode=mode,
        claims=claims,
        max_items=max_items,
        timeout_seconds=timeout_seconds,
        request_id=request_id,
        article_id=article_id,
        policy_version=policy.version,
    )
    return retrieve_evidence_request(
        request,
        provider=provider,
        policy=policy,
        api_key=api_key,
        corpus_config=corpus_config,
        corpus_store=corpus_store,
        corpus_embedding_provider=corpus_embedding_provider,
    )


def search_brave(
    query: EvidenceQuery,
    api_key: str,
    timeout_seconds: float,
    policy: TrustedSourcePolicy,
) -> list[EvidenceItem]:
    """Search Brave under one deadline with bounded, rate-aware retries."""

    if timeout_seconds <= 0:
        raise BraveTransportError("timeout", "Brave search deadline expired")
    deadline = time.monotonic() + float(timeout_seconds)
    params = urllib.parse.urlencode({"q": query.text, "count": "10", "search_lang": "en"})
    request = urllib.request.Request(
        f"{BRAVE_SEARCH_URL}?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
            "User-Agent": "bias-evidence-retriever/2.0",
        },
    )
    attempt = 0
    burst_retry_used = False
    network_retry_used = False
    diagnostics: dict[str, Any] = {"provider": "brave", "attempts": 0}
    while True:
        attempt += 1
        diagnostics["attempts"] = attempt
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BraveTransportError("timeout", "Brave search deadline expired", diagnostics=diagnostics)
        try:
            with urllib.request.urlopen(request, timeout=max(0.001, remaining)) as response:
                status = int(getattr(response, "status", 200) or 200)
                diagnostics = _brave_diagnostics(response.headers, status=status, attempts=attempt)
                body = response.read(2 * 1024 * 1024 + 1)
                if len(body) > 2 * 1024 * 1024:
                    raise BraveTransportError(
                        "invalid_response", "Brave response exceeded the size limit",
                        http_status=status, diagnostics=diagnostics,
                    )
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            diagnostics = _brave_diagnostics(exc.headers, status=status, attempts=attempt)
            retry_after = _retry_after_seconds(exc.headers)
            if status in {401, 403}:
                raise BraveTransportError(
                    "authentication", "Brave authentication failed",
                    http_status=status, diagnostics=diagnostics,
                ) from exc
            if status in {402}:
                raise BraveTransportError(
                    "quota_exhausted", "Brave quota is exhausted",
                    http_status=status, retry_after_seconds=retry_after, diagnostics=diagnostics,
                ) from exc
            if status == 429:
                quota_exhausted = _monthly_quota_exhausted(exc.headers)
                code = "quota_exhausted" if quota_exhausted else "rate_limit"
                delay = retry_after
                if delay is None:
                    delay = _shortest_rate_reset(exc.headers)
                remaining = deadline - time.monotonic()
                if (
                    not quota_exhausted
                    and not burst_retry_used
                    and delay is not None
                    and 0 <= delay < remaining
                ):
                    burst_retry_used = True
                    if delay:
                        time.sleep(delay)
                    continue
                raise BraveTransportError(
                    code,
                    "Brave quota is exhausted" if quota_exhausted else "Brave rate limit exceeded",
                    http_status=status,
                    retry_after_seconds=delay,
                    diagnostics=diagnostics,
                ) from exc
            if 500 <= status <= 599 and attempt < 3:
                delay = min(0.25 * (2 ** (attempt - 1)), 1.0)
                if delay < deadline - time.monotonic():
                    time.sleep(delay)
                    continue
            raise BraveTransportError(
                "provider_failure", "Brave provider returned an error",
                http_status=status, retry_after_seconds=retry_after, diagnostics=diagnostics,
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise BraveTransportError("timeout", "Brave search timed out", diagnostics=diagnostics) from exc
        except BraveTransportError:
            raise
        except (OSError, urllib.error.URLError) as exc:
            if not network_retry_used and 0.25 < deadline - time.monotonic():
                network_retry_used = True
                time.sleep(0.25)
                continue
            raise BraveTransportError(
                "network", "Brave search network failure", diagnostics=diagnostics
            ) from exc
        if not 200 <= status < 300:
            raise BraveTransportError(
                "provider_failure", "Brave provider returned an error",
                http_status=status, diagnostics=diagnostics,
            )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BraveTransportError(
                "invalid_response", "Brave returned invalid JSON",
                http_status=status, diagnostics=diagnostics,
            ) from exc
        if not isinstance(payload, dict):
            raise BraveTransportError(
                "invalid_response", "Brave returned an invalid response object",
                http_status=status, diagnostics=diagnostics,
            )
        web = payload.get("web", {})
        results = web.get("results", []) if isinstance(web, dict) else []
        if not isinstance(results, list):
            raise BraveTransportError(
                "invalid_response", "Brave returned invalid web results",
                http_status=status, diagnostics=diagnostics,
            )
        break

    evidence: list[EvidenceItem] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url") or "").strip()
        if not is_permitted_web_url(url, policy):
            continue
        source_type = classify_trusted_source(url, policy)
        if source_type is None:
            continue
        evidence.append(
            EvidenceItem(
                title=clean_text(str(result.get("title") or "")),
                url=url,
                domain=normalize_domain(url),
                source_type=source_type,
                published_date=extract_published_date(result),
                snippet=clean_text(str(result.get("description") or "")),
                query=query.text,
                retrieval_reason=query.source,
                claim_ids=query.claim_ids,
                canonical_url=canonicalize_web_url(url),
                retrieved_at=_utc_now(),
                provenance=EvidenceProvenance(
                    provider="brave",
                    query=query.text,
                    query_source=query.source,
                    collection=None,
                    document_id=None,
                    chunk_id=None,
                    point_id=None,
                    score=None,
                ),
                material_kind="search_snippet",
            )
        )
    diagnostics["result_count"] = len(evidence)
    return BraveSearchItems(evidence, diagnostics)


def _brave_diagnostics(headers: Any, *, status: int, attempts: int) -> dict[str, Any]:
    limits = _numeric_header_values(headers, "X-RateLimit-Limit")
    remaining = _numeric_header_values(headers, "X-RateLimit-Remaining")
    resets = _normalized_reset_values(headers, "X-RateLimit-Reset")
    policy = _safe_header(headers, "X-RateLimit-Policy")
    diagnostic: dict[str, Any] = {
        "provider": "brave",
        "http_status": status,
        "attempts": attempts,
    }
    if limits or remaining or resets or policy:
        diagnostic["rate_limit"] = {
            "limits": limits,
            "remaining": remaining,
            "reset_seconds": resets,
            "policy": policy,
        }
    retry_after = _retry_after_seconds(headers)
    if retry_after is not None:
        diagnostic["retry_after_seconds"] = retry_after
    return diagnostic


def _safe_header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except Exception:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9,;=._ -]", "", str(value or "")).strip()
    return cleaned[:500] or None


def _numeric_header_values(headers: Any, name: str) -> list[float]:
    raw = _safe_header(headers, name)
    if not raw:
        return []
    values: list[float] = []
    for part in raw.split(","):
        token = part.strip().split(";", 1)[0].strip()
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _normalized_reset_values(headers: Any, name: str) -> list[float]:
    now = time.time()
    values = _numeric_header_values(headers, name)
    return [max(0.0, value - now if value > 1_000_000_000 else value) for value in values]


def _retry_after_seconds(headers: Any) -> float | None:
    raw = _safe_header(headers, "Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _shortest_rate_reset(headers: Any) -> float | None:
    resets = _normalized_reset_values(headers, "X-RateLimit-Reset")
    return min(resets) if resets else None


def _monthly_quota_exhausted(headers: Any) -> bool:
    remaining = _numeric_header_values(headers, "X-RateLimit-Remaining")
    resets = _normalized_reset_values(headers, "X-RateLimit-Reset")
    if not remaining:
        return False
    if len(remaining) == 1:
        return remaining[0] <= 0 and bool(resets and max(resets) >= 86400)
    paired = list(zip(remaining, resets, strict=False))
    return any(value <= 0 and reset >= 86400 for value, reset in paired) or remaining[-1] <= 0


def extract_published_date(result: dict[str, Any]) -> str | None:
    for key in ("age", "page_age", "published", "date"):
        value = result.get(key)
        if value:
            return str(value)
    profile = result.get("profile")
    if isinstance(profile, dict) and profile.get("long_name"):
        return None
    return None


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _request_metadata(
    request: EvidenceRequest,
    policy: TrustedSourcePolicy,
    provider: str,
) -> dict[str, Any]:
    normalized_mode = request.mode.casefold()
    return {
        "requested_mode": request.mode,
        "effective_mode": (
            normalized_mode
            if normalized_mode in {"off", "web", "corpus", "hybrid"}
            else "off"
        ),
        "request_id": request.request_id,
        "article_id": request.article_id,
        "policy_version": request.policy_version or policy.version,
        "provider": provider,
        "queries": [],
    }


def _result(
    *,
    status: str,
    items: list[EvidenceItem],
    metadata: dict[str, Any],
    started: float,
    error: str | None = None,
    failures: list[EvidenceFailure] | None = None,
    provider_diagnostics: list[dict[str, Any]] | None = None,
) -> EvidenceResult:
    return EvidenceResult(
        status=status,
        items=items,
        error=error,
        requested_mode=metadata["requested_mode"],
        effective_mode=metadata["effective_mode"],
        elapsed_ms=_elapsed_ms(started),
        request_id=metadata["request_id"],
        article_id=metadata["article_id"],
        policy_version=metadata["policy_version"],
        provider=metadata["provider"],
        queries=metadata["queries"],
        failures=failures or [],
        provider_diagnostics=provider_diagnostics or metadata.get("provider_diagnostics", []),
    )


def _error_result(
    *,
    metadata: dict[str, Any],
    started: float,
    error: str,
    failure: EvidenceFailure,
    effective_mode: str | None = None,
) -> EvidenceResult:
    if effective_mode is not None:
        metadata = dict(metadata)
        metadata["effective_mode"] = effective_mode
    return _result(
        status="error",
        items=[],
        error=error,
        metadata=metadata,
        started=started,
        failures=[failure],
    )


def _merge_ids(*id_groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for ids in id_groups:
        for value in ids:
            if value not in seen:
                seen.add(value)
                merged.append(value)
    return tuple(merged)


def _provider_failure(
    exc: Exception,
    *,
    provider: str,
    stage: str,
    message: str,
    query: str | None = None,
    query_source: str | None = None,
) -> EvidenceFailure:
    code = getattr(exc, "code", None)
    http_status = getattr(exc, "http_status", None)
    retry_after = getattr(exc, "retry_after_seconds", None)
    if not code:
        if isinstance(exc, (TimeoutError, socket.timeout)):
            code = "timeout"
        elif isinstance(exc, (OSError, urllib.error.URLError)):
            code = "network"
        else:
            code = "provider_failure"
    return EvidenceFailure(
        provider=provider,
        stage=stage,
        message=message,
        query=query,
        query_source=query_source,
        exception_type=type(exc).__name__,
        code=str(code),
        http_status=int(http_status) if http_status is not None else None,
        retry_after_seconds=float(retry_after) if retry_after is not None else None,
    )


def _cosine_similarity(left: Any, right: Any) -> float:
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if len(left_values) != len(right_values) or not left_values:
        raise ValueError("Embedding vectors must be non-empty and have equal dimensions")
    numerator = sum(a * b for a, b in zip(left_values, right_values, strict=True))
    denominator = math.sqrt(sum(value * value for value in left_values)) * math.sqrt(
        sum(value * value for value in right_values)
    )
    return numerator / denominator if denominator else 0.0


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
