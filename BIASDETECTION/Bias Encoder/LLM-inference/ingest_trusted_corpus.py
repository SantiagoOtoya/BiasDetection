#!/usr/bin/env python
"""Ingest explicitly supplied, policy-permitted public documents into Qdrant.

The command is intentionally not a crawler: every run receives one URL or one
reviewed local HTML/text file.  It never ingests a source merely because that
source appears in the registry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping, Sequence

import retrieval_store
import runtime_config
import source_fetch
import trusted_source_registry


DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_FETCH_TIMEOUT_SECONDS = 15.0
SUPPORTED_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}
SUPPORTED_FILE_SUFFIXES = {".html", ".htm", ".txt"}


class IngestionInputError(ValueError):
    """The supplied content cannot safely be ingested."""


class SourcePolicyError(IngestionInputError):
    """The supplied source does not meet the trusted-corpus policy."""


class OptionalDependencyError(RuntimeError):
    """A dependency used only by the ingestion command is unavailable."""


# Version 2 is shared by inference-time fetching and corpus operations.  Keep
# these names exported here for callers of the original ingestion module.
SourceRule = trusted_source_registry.SourceRule
SourcePolicyDecision = trusted_source_registry.SourcePolicyDecision
TrustedSourceRegistry = trusted_source_registry.TrustedSourceRegistry


@dataclass(frozen=True)
class FetchedContent:
    source_url: str
    content_type: str
    charset: str | None
    body: bytes
    fetched_at: str


@dataclass(frozen=True)
class ExtractedContent:
    text: str
    title: str | None
    publication_candidates: tuple[str, ...]
    canonical_href: str | None


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class _DocumentHTMLParser(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "canvas"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.title_parts: list[str] = []
        self.publication_candidates: list[str] = []
        self.canonical_href: str | None = None
        self._skip_depth = 0
        self._in_title = False
        self._jsonld_depth = 0
        self._jsonld_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        attributes = {str(key).casefold(): value or "" for key, value in attrs}
        if tag == "title":
            self._in_title = True
        if tag == "link":
            rel = attributes.get("rel", "").casefold().split()
            if "canonical" in rel and attributes.get("href"):
                self.canonical_href = attributes["href"].strip()
        if tag == "meta":
            key = (
                attributes.get("property")
                or attributes.get("name")
                or attributes.get("itemprop")
                or ""
            ).casefold()
            value = attributes.get("content", "").strip()
            if key in {
                "article:published_time",
                "datepublished",
                "date",
                "parsely-pub-date",
                "publishdate",
            } and value:
                self.publication_candidates.append(value)
        if tag == "script" and attributes.get("type", "").casefold() == "application/ld+json":
            self._jsonld_depth += 1
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "title":
            self._in_title = False
        if tag == "script" and self._jsonld_depth:
            self._jsonld_depth -= 1
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._jsonld_depth:
            self._jsonld_parts.append(data)
            return
        if self._in_title:
            self.title_parts.append(data)
        if not self._skip_depth:
            self.text_parts.append(data)

    def extracted(self) -> ExtractedContent:
        for raw in self._jsonld_parts:
            try:
                self.publication_candidates.extend(_jsonld_publication_dates(json.loads(raw)))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return ExtractedContent(
            text=retrieval_store.normalise_text(" ".join(self.text_parts)),
            title=retrieval_store.normalise_text(" ".join(self.title_parts)) or None,
            publication_candidates=tuple(self.publication_candidates),
            canonical_href=self.canonical_href,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is not None and list(argv[:1]) == ["single"]:
        argv = list(argv)[1:]
    parser = argparse.ArgumentParser(
        description="Ingest one explicitly permitted public HTML/text document into optional Qdrant storage.",
        epilog=(
            "Lifecycle commands are also available as the first argument: readiness, "
            "manifest-validate, manifest-approve, batch, refresh, expiry, stats, "
            "health, backup, restore-verify, scheduler, and live-test. See "
            "CORPUS-OPERATIONS.md."
        ),
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url", help="Permitted HTTPS document URL to fetch")
    source_group.add_argument("--file", type=Path, help="Reviewed local .html, .htm, or .txt file")
    parser.add_argument(
        "--source-url",
        help="Required permitted HTTPS provenance URL when --file is used",
    )
    parser.add_argument("--title", help="Optional title override")
    parser.add_argument("--published-at", help="Optional ISO-8601 publication timestamp")
    parser.add_argument(
        "--trusted-sources-file",
        type=Path,
        default=Path(__file__).with_name("trusted_sources.json"),
        help="Trusted-source registry JSON file",
    )
    parser.add_argument("--chunk-words", type=int, default=retrieval_store.DEFAULT_CHUNK_WORDS)
    parser.add_argument(
        "--chunk-overlap-words",
        type=int,
        default=retrieval_store.DEFAULT_CHUNK_OVERLAP_WORDS,
    )
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--fetch-timeout-seconds", type=float, default=DEFAULT_FETCH_TIMEOUT_SECONDS)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Explicit dotenv file; existing process variables take precedence",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the reviewed Qdrant mutation (default is a non-mutating dry-run)",
    )
    parser.add_argument("--rights-reviewer", help="Required reviewer identity for --apply")
    parser.add_argument("--rights-reviewed-at", help="Required ISO-8601 review date for --apply")
    parser.add_argument(
        "--third-party-marked",
        action="store_true",
        help="Declare third-party-marked material (always rejects ingestion)",
    )

    parser.add_argument("--qdrant-local-path")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--qdrant-api-key")
    parser.add_argument("--collection")
    parser.add_argument("--embedding-model")
    parser.add_argument("--vector-name")
    parser.add_argument("--content-field")
    parser.add_argument("--qdrant-timeout-seconds", type=float)
    return parser.parse_args(argv)


def build_qdrant_config(args: argparse.Namespace) -> retrieval_store.QdrantCorpusConfig:
    base = retrieval_store.QdrantCorpusConfig.from_environment()
    local_path = base.local_path
    url = base.url
    api_key = base.api_key
    if args.qdrant_local_path is not None:
        local_path = args.qdrant_local_path
        url = None
        api_key = None
    if args.qdrant_url is not None:
        url = args.qdrant_url
        local_path = None
    if args.qdrant_api_key is not None:
        api_key = args.qdrant_api_key
    return retrieval_store.QdrantCorpusConfig(
        local_path=local_path,
        url=url,
        api_key=api_key,
        collection=args.collection or base.collection,
        embedding_model=args.embedding_model or base.embedding_model,
        vector_name=args.vector_name or base.vector_name,
        content_field=args.content_field or base.content_field,
        timeout_seconds=args.qdrant_timeout_seconds or base.timeout_seconds,
    )


def fetch_remote_content(
    url: str,
    registry: TrustedSourceRegistry,
    *,
    timeout_seconds: float,
    max_bytes: int,
) -> FetchedContent:
    contact = str(os.environ.get("CORPUS_FETCH_CONTACT") or "").strip()
    try:
        extracted = source_fetch.fetch_source_document(
            url,
            registry,
            contact=contact,
            deadline=time.monotonic() + timeout_seconds,
            max_bytes=max_bytes,
        )
    except source_fetch.SourceFetchError as exc:
        if exc.code == "policy_rejected":
            raise SourcePolicyError(str(exc)) from exc
        raise IngestionInputError(str(exc)) from exc
    # Compatibility return type: controlled extraction has already reduced the
    # remote source to normalized text, so callers cannot accidentally bypass it.
    return FetchedContent(
        source_url=extracted.canonical_url,
        content_type="text/plain",
        charset="utf-8",
        body=extracted.text.encode("utf-8"),
        fetched_at=extracted.fetched_at,
    )


def load_local_content(
    path: Path,
    source_url: str,
    *,
    max_bytes: int,
) -> FetchedContent:
    if path.suffix.casefold() not in SUPPORTED_FILE_SUFFIXES:
        raise IngestionInputError("Local files must be .html, .htm, or .txt")
    try:
        if path.stat().st_size > max_bytes:
            raise IngestionInputError("Local file exceeds the configured maximum document size")
        body = path.read_bytes()
    except OSError as exc:
        raise IngestionInputError(f"Could not read local source file: {path}") from exc
    content_type = "text/plain" if path.suffix.casefold() == ".txt" else "text/html"
    canonical_url = retrieval_store.canonicalize_document_url(source_url)
    if canonical_url is None:
        raise SourcePolicyError("Local file provenance must be an absolute HTTPS URL")
    return FetchedContent(
        source_url=canonical_url,
        content_type=content_type,
        charset="utf-8",
        body=body,
        fetched_at=retrieval_store.utc_now(),
    )


def document_from_content(
    fetched: FetchedContent,
    registry: TrustedSourceRegistry,
    *,
    title_override: str | None = None,
    published_at_override: str | None = None,
) -> retrieval_store.CorpusDocument:
    source_decision = registry.evaluate_url(fetched.source_url)
    if not source_decision.allowed or source_decision.rule is None:
        raise SourcePolicyError(source_decision.reason or "source policy rejected URL")
    decoded = _decode_content(fetched.body, fetched.charset)
    extracted = extract_content(decoded, fetched.content_type)
    canonical_decision = source_decision
    canonical_url = source_decision.canonical_url or fetched.source_url
    if extracted.canonical_href:
        candidate = urllib.parse.urljoin(fetched.source_url, extracted.canonical_href)
        candidate_decision = registry.evaluate_url(candidate)
        if candidate_decision.allowed and candidate_decision.rule is not None:
            canonical_decision = candidate_decision
            canonical_url = candidate_decision.canonical_url or canonical_url
    rule = canonical_decision.rule
    assert rule is not None
    publication_candidates = (
        (published_at_override,) if published_at_override else ()
    ) + extracted.publication_candidates
    published_at = _first_normalised_date(publication_candidates)
    source = retrieval_store.SourceMetadata(
        source_id=rule.source_id,
        source_name=rule.source_name,
        source_url=source_decision.canonical_url or fetched.source_url,
        canonical_url=canonical_url,
        source_domain=retrieval_store.normalise_domain(canonical_url),
        source_tier=rule.source_tier,
        source_policy_version=registry.policy_version,
        license_id=rule.license_id,
        license_url=rule.license_url,
        redistribution_allowed=rule.redistribution_allowed,
        retention_policy=rule.retention_policy,
        retention_days=rule.retention_days,
        ingest_permitted=True,
    )
    return retrieval_store.CorpusDocument(
        title=title_override or extracted.title or canonical_url,
        text=extracted.text,
        source=source,
        published_at=published_at,
        fetched_at=fetched.fetched_at,
    )


def extract_content(text: str, content_type: str) -> ExtractedContent:
    if content_type in {"text/html", "application/xhtml+xml"}:
        parser = _DocumentHTMLParser()
        try:
            parser.feed(text)
            parser.close()
        except Exception as exc:
            raise IngestionInputError("Could not parse HTML source") from exc
        extracted = parser.extracted()
    elif content_type == "text/plain":
        extracted = ExtractedContent(
            text=retrieval_store.normalise_text(text),
            title=None,
            publication_candidates=(),
            canonical_href=None,
        )
    else:
        raise IngestionInputError(f"Unsupported document content type: {content_type}")
    if not extracted.text:
        raise IngestionInputError("No extractable text was found in the source")
    return extracted


class SentenceTransformerEmbedder:
    """Small adapter so tests can inject an in-memory embedding provider."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OptionalDependencyError(
                "sentence-transformers is required for corpus ingestion"
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
            dimension = self._model.get_sentence_embedding_dimension()
        except Exception as exc:
            raise OptionalDependencyError("Could not load the configured embedding model") from exc
        if dimension is None or int(dimension) <= 0:
            raise OptionalDependencyError("Embedding model did not provide a valid vector dimension")
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        try:
            vectors = self._model.encode(
                list(texts), normalize_embeddings=True, show_progress_bar=False
            )
            return [[float(value) for value in vector] for vector in vectors]
        except Exception as exc:
            raise OptionalDependencyError("Could not generate corpus embeddings") from exc


def run_ingestion(args: argparse.Namespace) -> retrieval_store.CorpusIngestionResult:
    if args.max_bytes <= 0:
        raise IngestionInputError("--max-bytes must be greater than zero")
    registry = TrustedSourceRegistry.from_file(args.trusted_sources_file)
    if args.url:
        if args.source_url:
            raise IngestionInputError("--source-url is only valid with --file")
        initial_decision = registry.evaluate_url(args.url)
        if not initial_decision.allowed:
            raise SourcePolicyError(initial_decision.reason or "source policy rejected URL")
    else:
        if not args.source_url:
            raise IngestionInputError("--source-url is required with --file")
        initial_decision = registry.evaluate_url(args.source_url)
        if not initial_decision.allowed:
            raise SourcePolicyError(initial_decision.reason or "source policy rejected URL")

    config = build_qdrant_config(args)
    if args.apply and config.mode != "remote":
        raise IngestionInputError("Applied corpus mutations require Qdrant Cloud via QDRANT_URL")
    if args.apply and not config.api_key:
        raise IngestionInputError("QDRANT_API_KEY is required for applied corpus mutations")
    if not args.apply:
        return retrieval_store.CorpusIngestionResult(
            status="dry_run",
            collection=config.collection,
            reason="Validated source policy; rerun with --apply to fetch, embed, and mutate Qdrant",
        )
    registry.require_document_rights(
        initial_decision.canonical_url or args.url or args.source_url,
        reviewer=args.rights_reviewer,
        reviewed_at=args.rights_reviewed_at,
        third_party_marked=args.third_party_marked,
    )
    store = retrieval_store.QdrantCorpusStore(config)
    health = store.health()
    if health.status != "ready":
        return retrieval_store.CorpusIngestionResult(
            status=health.status,
            collection=config.collection,
            reason=health.reason,
        )

    if args.url:
        contact = str(os.environ.get("CORPUS_FETCH_CONTACT") or "").strip()
        if not contact:
            raise IngestionInputError("CORPUS_FETCH_CONTACT is required for source fetching")
        extracted = source_fetch.fetch_source_document(
            args.url,
            registry,
            contact=contact,
            deadline=time.monotonic() + args.fetch_timeout_seconds,
            max_bytes=args.max_bytes,
        )
        decision = registry.evaluate_url(extracted.canonical_url, purpose="ingest")
        if not decision.allowed or decision.rule is None:
            raise SourcePolicyError(decision.reason or "source policy rejected extracted URL")
        rule = decision.rule
        source_fetch.stage_source_document(
            extracted,
            rule,
            staging_dir=Path(__file__).with_name(".corpus") / "staging",
        )
        document = retrieval_store.CorpusDocument(
            title=args.title or extracted.title,
            text=extracted.text,
            source=retrieval_store.SourceMetadata(
                source_id=rule.source_id,
                source_name=rule.source_name,
                source_url=extracted.source_url,
                canonical_url=extracted.canonical_url,
                source_domain=retrieval_store.normalise_domain(extracted.canonical_url),
                source_tier=rule.source_tier,
                source_policy_version=registry.policy_version,
                license_id=rule.license_id,
                license_url=rule.license_url,
                redistribution_allowed=rule.redistribution_allowed,
                retention_policy=rule.retention_policy,
                retention_days=rule.retention_days,
                ingest_permitted=True,
            ),
            published_at=args.published_at or extracted.published_at,
            fetched_at=extracted.fetched_at,
        )
    else:
        fetched = load_local_content(args.file, args.source_url, max_bytes=args.max_bytes)
        document = document_from_content(
            fetched,
            registry,
            title_override=args.title,
            published_at_override=args.published_at,
        )
    embedder = SentenceTransformerEmbedder(config.embedding_model)
    chunks = retrieval_store.chunk_document_text(
        document.text,
        chunk_words=args.chunk_words,
        chunk_overlap_words=args.chunk_overlap_words,
    )
    vectors = embedder.embed([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks) or any(len(vector) != embedder.dimension for vector in vectors):
        raise OptionalDependencyError("Embedding model returned vectors with an unexpected dimension")
    return store.ingest_document(
        document,
        vectors,
        chunk_words=args.chunk_words,
        chunk_overlap_words=args.chunk_overlap_words,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "single":
        arguments = arguments[1:]
    if arguments and arguments[0] in {
        "readiness", "manifest-validate", "manifest-approve", "batch", "refresh",
        "expiry", "stats", "health", "backup", "restore-verify", "scheduler",
        "weekly-maintenance", "monthly-restore-verify", "health-statistics", "live-test",
    }:
        import corpus_operations

        return corpus_operations.main(arguments)
    args = parse_args(arguments)
    try:
        runtime_config.load_explicit_env_file(args.env_file)
        result = run_ingestion(args)
    except SourcePolicyError as exc:
        _write_result({"status": "rejected_policy", "reason": str(exc)})
        return 2
    except (
        IngestionInputError,
        OptionalDependencyError,
        retrieval_store.CorpusConfigurationError,
        trusted_source_registry.SourceRegistryError,
        runtime_config.RuntimeConfigurationError,
    ) as exc:
        _write_result({"status": "rejected_input", "reason": str(exc)})
        return 2
    except Exception:
        _write_result({"status": "error", "reason": "Corpus ingestion failed"})
        return 3
    _write_result(result.to_json())
    return 0 if result.status in {"dry_run", "ingested", "replaced", "duplicate"} else 3


def _normalise_allowed_prefix(url: Any) -> str | None:
    canonical = retrieval_store.canonicalize_document_url(str(url or ""))
    return canonical if canonical else None


def _parse_content_type(header: str | None) -> tuple[str, str | None]:
    raw = str(header or "").strip()
    mime, _, parameters = raw.partition(";")
    content_type = mime.strip().casefold()
    charset = None
    for parameter in parameters.split(";"):
        key, separator, value = parameter.partition("=")
        if separator and key.strip().casefold() == "charset":
            charset = value.strip().strip('"') or None
    return content_type, charset


def _decode_content(body: bytes, charset: str | None) -> str:
    encodings = [charset] if charset else []
    encodings.extend(["utf-8", "utf-8-sig", "latin-1"])
    for encoding in encodings:
        if not encoding:
            continue
        try:
            return body.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


def _jsonld_publication_dates(value: Any) -> list[str]:
    dates: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in {"datepublished", "datecreated"} and isinstance(child, str):
                dates.append(child)
            dates.extend(_jsonld_publication_dates(child))
    elif isinstance(value, list):
        for child in value:
            dates.extend(_jsonld_publication_dates(child))
    return dates


def _first_normalised_date(values: Sequence[str | None]) -> str | None:
    for value in values:
        if not value:
            continue
        try:
            return retrieval_store.normalise_datetime(value)
        except (TypeError, ValueError):
            continue
    return None


def _write_result(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
