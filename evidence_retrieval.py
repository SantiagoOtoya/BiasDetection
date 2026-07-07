#!/usr/bin/env python
"""Retrieve strictly trusted evidence for bias-report context windows."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_MAX_QUERIES = 3
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
    wire_factual_domains: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceQuery:
    text: str
    source: str


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
    supports_claim: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceResult:
    status: str
    items: list[EvidenceItem]
    error: str | None = None

    def items_json(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index, item in enumerate(self.items, start=1):
            record = item.to_json()
            record["citation_id"] = f"E{index}"
            records.append(record)
        return records


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
    wire_factual_domains=(
        "apnews.com",
        "reuters.com",
    ),
)


def load_trusted_source_policy(path: Path | str | None) -> TrustedSourcePolicy:
    if path is None:
        return DEFAULT_TRUSTED_SOURCE_POLICY

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrustedSourcePolicy(
        official_suffixes=tuple(data.get("official_suffixes", ())),
        primary_official_domains=tuple(data.get("primary_official_domains", ())),
        empirical_research_domains=tuple(data.get("empirical_research_domains", ())),
        wire_factual_domains=tuple(data.get("wire_factual_domains", ())),
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
    if any(domain_matches(domain, item) for item in policy.wire_factual_domains):
        return "wire_factual"
    if any(domain.endswith(suffix) for suffix in policy.official_suffixes):
        return "primary_official"
    if any(domain_matches(domain, item) for item in policy.primary_official_domains):
        return "primary_official"
    return None


def build_evidence_queries(
    selected_sentence_texts: list[str],
    context_text: str,
    max_queries: int = DEFAULT_MAX_QUERIES,
) -> list[EvidenceQuery]:
    queries: list[EvidenceQuery] = []
    for sentence in selected_sentence_texts:
        query_text = compact_query(sentence)
        if query_text:
            queries.append(EvidenceQuery(text=query_text, source="selected_sentence"))
        if len(queries) >= max_queries:
            return dedupe_queries(queries)

    context_query = compact_query(context_text)
    if context_query:
        queries.append(EvidenceQuery(text=context_query, source="context_window"))
    return dedupe_queries(queries)[:max_queries]


def compact_query(text: str, max_terms: int = 14) -> str:
    words = [
        match.group(0)
        for match in SIGNIFICANT_WORD_RE.finditer(text.replace("’", "'"))
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
    seen: set[str] = set()
    for query in queries:
        key = query.text.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)
    return deduped


def retrieve_evidence(
    selected_sentence_texts: list[str],
    context_text: str,
    provider: str,
    max_items: int,
    timeout_seconds: float,
    policy: TrustedSourcePolicy = DEFAULT_TRUSTED_SOURCE_POLICY,
    api_key: str | None = None,
) -> EvidenceResult:
    if max_items <= 0:
        return EvidenceResult(status="none_found", items=[])
    if provider != "brave":
        return EvidenceResult(status="error", items=[], error=f"Unsupported evidence provider: {provider}")

    queries = build_evidence_queries(selected_sentence_texts, context_text)
    if not queries:
        return EvidenceResult(status="none_found", items=[])

    token = api_key or os.environ.get("BRAVE_SEARCH_API_KEY")
    if not token:
        return EvidenceResult(
            status="error",
            items=[],
            error="BRAVE_SEARCH_API_KEY is not set",
        )

    items: list[EvidenceItem] = []
    seen_urls: set[str] = set()
    try:
        for query in queries:
            for item in search_brave(
                query=query,
                api_key=token,
                timeout_seconds=timeout_seconds,
                policy=policy,
            ):
                if item.url in seen_urls:
                    continue
                seen_urls.add(item.url)
                items.append(item)
                if len(items) >= max_items:
                    return EvidenceResult(status="found", items=items)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return EvidenceResult(status="error", items=items, error=str(exc))

    if items:
        return EvidenceResult(status="found", items=items)
    return EvidenceResult(status="none_found", items=[])


def search_brave(
    query: EvidenceQuery,
    api_key: str,
    timeout_seconds: float,
    policy: TrustedSourcePolicy,
) -> list[EvidenceItem]:
    params = urllib.parse.urlencode({"q": query.text, "count": "10", "search_lang": "en"})
    request = urllib.request.Request(
        f"{BRAVE_SEARCH_URL}?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
            "User-Agent": "bias-evidence-retriever/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    results = payload.get("web", {}).get("results", [])
    evidence: list[EvidenceItem] = []
    for result in results:
        url = str(result.get("url") or "").strip()
        source_type = classify_trusted_source(url, policy)
        if source_type is None:
            continue
        domain = normalize_domain(url)
        evidence.append(
            EvidenceItem(
                title=clean_text(str(result.get("title") or "")),
                url=url,
                domain=domain,
                source_type=source_type,
                published_date=extract_published_date(result),
                snippet=clean_text(str(result.get("description") or "")),
                query=query.text,
                retrieval_reason=query.source,
            )
        )
    return evidence


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
