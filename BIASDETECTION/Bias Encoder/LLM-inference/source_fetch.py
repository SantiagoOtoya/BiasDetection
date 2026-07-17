#!/usr/bin/env python
"""Deadline-bound, SSRF-safe extraction and content-addressed staging."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping

import retrieval_store
from trusted_source_registry import SourceRule, TrustedSourceRegistry


DEFAULT_MAX_BYTES = 12 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_MAX_PDF_PAGES = 250
HTML_EXTRACTION_VERSION = "trafilatura-main-text/v1"
PDF_EXTRACTION_VERSION = "pypdf-digital-text/v1"
TEXT_EXTRACTION_VERSION = "plain-text/v1"
SUPPORTED_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "application/pdf",
}


class SourceFetchError(RuntimeError):
    """A source cannot be fetched or extracted safely."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


@dataclass(frozen=True)
class ExtractedSourceDocument:
    source_url: str
    canonical_url: str
    title: str
    text: str
    content_type: str
    fetched_at: str
    extraction_version: str
    archive_hash: str
    content_hash: str
    etag: str | None = None
    last_modified: str | None = None
    published_at: str | None = None
    page_count: int | None = None
    archive_bytes: bytes = field(default=b"", repr=False, compare=False)


@dataclass(frozen=True)
class StagedSourceRecord:
    schema_version: str
    stage_id: str
    source_id: str
    canonical_url: str
    content_hash: str
    archive_hash: str
    extraction_version: str
    metadata_path: str
    text_path: str
    rights_status: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_title = False
        self.title_parts: list[str] = []
        self.canonical_href: str | None = None
        self.publication_dates: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {str(key).casefold(): str(value or "") for key, value in attrs}
        normalized = tag.casefold()
        if normalized == "title":
            self.in_title = True
        elif normalized == "link" and "canonical" in attributes.get("rel", "").casefold().split():
            self.canonical_href = attributes.get("href") or self.canonical_href
        elif normalized == "meta":
            key = (
                attributes.get("property")
                or attributes.get("name")
                or attributes.get("itemprop")
                or ""
            ).casefold()
            if key in {"article:published_time", "datepublished", "date", "publishdate"}:
                value = attributes.get("content", "").strip()
                if value:
                    self.publication_dates.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)


def fetch_source_document(
    url: str,
    registry: TrustedSourceRegistry,
    *,
    contact: str,
    deadline: float,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
    opener: Any | None = None,
    resolver: Callable[..., Any] = socket.getaddrinfo,
) -> ExtractedSourceDocument:
    """Fetch one allow-listed URL while validating every network transition."""

    contact_value = str(contact or "").strip()
    if not contact_value or "\n" in contact_value or "\r" in contact_value:
        raise SourceFetchError("configuration", "CORPUS_FETCH_CONTACT is required")
    if max_bytes <= 0 or max_redirects < 0:
        raise ValueError("Fetch size and redirect bounds must be valid")
    current_url = url
    client = opener or urllib.request.build_opener(_NoRedirectHandler())
    for redirect_count in range(max_redirects + 1):
        remaining = _remaining(deadline)
        decision = registry.evaluate_url(current_url, purpose="fetch")
        if not decision.allowed or decision.canonical_url is None:
            raise SourceFetchError("policy_rejected", decision.reason or "Source policy rejected URL")
        _require_public_https(decision.canonical_url, resolver=resolver, deadline=deadline)
        request = urllib.request.Request(
            decision.canonical_url,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9",
                "User-Agent": f"bias-trusted-corpus-fetch/2.0 ({contact_value})",
                "From": contact_value,
            },
        )
        try:
            response = client.open(request, timeout=remaining)
        except urllib.error.HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                if redirect_count >= max_redirects:
                    raise SourceFetchError("redirect_limit", "Source exceeded redirect limit") from exc
                location = exc.headers.get("Location")
                if not location:
                    raise SourceFetchError("invalid_redirect", "Redirect omitted Location") from exc
                current_url = urllib.parse.urljoin(decision.canonical_url, location)
                continue
            raise SourceFetchError(
                "http_error", f"Source returned HTTP {exc.code}", http_status=int(exc.code)
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise SourceFetchError("timeout", "Source fetch deadline expired") from exc
        except (OSError, urllib.error.URLError) as exc:
            raise SourceFetchError("network_error", "Source fetch network failure") from exc

        with response:
            _remaining(deadline)
            status = int(getattr(response, "status", 200) or 200)
            if not 200 <= status < 300:
                raise SourceFetchError("http_error", f"Source returned HTTP {status}", http_status=status)
            final_url = str(response.geturl() or decision.canonical_url)
            final_decision = registry.evaluate_url(final_url, purpose="fetch")
            if not final_decision.allowed or final_decision.canonical_url is None:
                raise SourceFetchError(
                    "policy_rejected", final_decision.reason or "Final response URL was rejected"
                )
            _require_public_https(final_decision.canonical_url, resolver=resolver, deadline=deadline)
            content_type, charset = parse_content_type(response.headers.get("Content-Type"))
            if content_type not in SUPPORTED_CONTENT_TYPES:
                raise SourceFetchError(
                    "unsupported_content_type",
                    f"Unsupported source content type: {content_type or 'unspecified'}",
                )
            length = response.headers.get("Content-Length")
            try:
                if length and int(length) > max_bytes:
                    raise SourceFetchError("oversized", "Source exceeds maximum size")
            except ValueError as exc:
                raise SourceFetchError("invalid_response", "Invalid Content-Length header") from exc
            body = response.read(max_bytes + 1)
            _remaining(deadline)
            if len(body) > max_bytes:
                raise SourceFetchError("oversized", "Source exceeds maximum size")
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="source-extract")
            future = executor.submit(
                extract_source_document,
                body,
                content_type=content_type,
                charset=charset,
                source_url=final_decision.canonical_url,
                registry=registry,
                fetched_at=retrieval_store.utc_now(),
                etag=_clean_header(response.headers.get("ETag")),
                last_modified=_clean_header(response.headers.get("Last-Modified")),
                max_pdf_pages=max_pdf_pages,
            )
            try:
                return future.result(timeout=_remaining(deadline))
            except FutureTimeoutError as exc:
                future.cancel()
                raise SourceFetchError("timeout", "Source extraction exceeded the deadline") from exc
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
    raise SourceFetchError("redirect_limit", "Source exceeded redirect limit")


def extract_source_document(
    body: bytes,
    *,
    content_type: str,
    charset: str | None,
    source_url: str,
    registry: TrustedSourceRegistry,
    fetched_at: str,
    etag: str | None = None,
    last_modified: str | None = None,
    max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
) -> ExtractedSourceDocument:
    archive_hash = hashlib.sha256(body).hexdigest()
    canonical_url = source_url
    title = source_url
    published_at = None
    page_count: int | None = None
    if content_type in {"text/html", "application/xhtml+xml"}:
        decoded = decode_bytes(body, charset)
        parser = _MetadataParser()
        try:
            parser.feed(decoded)
            parser.close()
            from trafilatura import extract
        except ImportError as exc:
            raise SourceFetchError(
                "dependency_missing", "Trafilatura is required for HTML extraction"
            ) from exc
        except Exception as exc:
            raise SourceFetchError("extraction_failed", "Could not parse source HTML") from exc
        try:
            text = extract(
                decoded,
                url=source_url,
                output_format="txt",
                include_comments=False,
                include_tables=True,
                include_links=False,
                favor_precision=True,
            )
        except Exception as exc:
            raise SourceFetchError("extraction_failed", "Trafilatura extraction failed") from exc
        title = retrieval_store.normalise_text(" ".join(parser.title_parts)) or source_url
        if parser.canonical_href:
            candidate = urllib.parse.urljoin(source_url, parser.canonical_href)
            decision = registry.evaluate_url(candidate, purpose="fetch")
            if decision.allowed and decision.canonical_url:
                canonical_url = decision.canonical_url
        published_at = _first_date(parser.publication_dates)
        extraction_version = HTML_EXTRACTION_VERSION
    elif content_type == "application/pdf":
        text, page_count = extract_pdf_text(body, max_pages=max_pdf_pages)
        extraction_version = PDF_EXTRACTION_VERSION
    elif content_type == "text/plain":
        text = decode_bytes(body, charset)
        extraction_version = TEXT_EXTRACTION_VERSION
    else:
        raise SourceFetchError("unsupported_content_type", "Unsupported source content type")
    normalized = retrieval_store.normalise_text(str(text or ""))
    if not normalized:
        raise SourceFetchError("no_text", "Source contained no extractable text")
    if len(normalized) < 120:
        raise SourceFetchError("low_quality", "Extracted source text is too short")
    return ExtractedSourceDocument(
        source_url=source_url,
        canonical_url=canonical_url,
        title=title,
        text=normalized,
        content_type=content_type,
        fetched_at=fetched_at,
        extraction_version=extraction_version,
        archive_hash=archive_hash,
        content_hash=retrieval_store.sha256_text(normalized),
        etag=etag,
        last_modified=last_modified,
        published_at=published_at,
        page_count=page_count,
        archive_bytes=body,
    )


def extract_pdf_text(body: bytes, *, max_pages: int = DEFAULT_MAX_PDF_PAGES) -> tuple[str, int]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SourceFetchError("dependency_missing", "pypdf is required for PDF extraction") from exc
    try:
        reader = PdfReader(BytesIO(body), strict=True)
    except Exception as exc:
        raise SourceFetchError("invalid_pdf", "Could not parse PDF") from exc
    if bool(getattr(reader, "is_encrypted", False)):
        raise SourceFetchError("encrypted_pdf", "Encrypted PDFs are not accepted")
    page_count = len(reader.pages)
    if page_count <= 0:
        raise SourceFetchError("no_text", "PDF contains no pages")
    if page_count > max_pages:
        raise SourceFetchError("oversized_pdf", "PDF exceeds the configured page limit")
    parts: list[str] = []
    pages_with_text = 0
    try:
        for page in reader.pages:
            value = retrieval_store.normalise_text(page.extract_text() or "")
            if value:
                pages_with_text += 1
                parts.append(value)
    except Exception as exc:
        raise SourceFetchError("invalid_pdf", "PDF text extraction failed") from exc
    text = retrieval_store.normalise_text(" ".join(parts))
    letters = sum(character.isalpha() for character in text)
    if (
        not text
        or pages_with_text == 0
        or len(text) < max(200, page_count * 20)
        or letters / max(1, len(text)) < 0.35
    ):
        raise SourceFetchError(
            "image_only_pdf", "PDF lacks sufficient digital text; OCR is not supported"
        )
    return text, page_count


def stage_source_document(
    document: ExtractedSourceDocument,
    rule: SourceRule,
    *,
    staging_dir: Path | str,
) -> StagedSourceRecord:
    """Write an immutable candidate; this function never writes to Qdrant."""

    if not rule.ingest_allowed or not rule.redistribution_allowed:
        raise SourceFetchError("rights_rejected", "Source content is not eligible for staging")
    root = Path(staging_dir)
    root.mkdir(parents=True, exist_ok=True)
    stage_id = f"stage_{document.content_hash}"
    document_id = retrieval_store.stable_document_id(document.canonical_url)
    chunks = retrieval_store.chunk_document_text(document.text)
    chunk_ids = [
        retrieval_store.stable_chunk_id(
            document_id,
            chunk.index,
            retrieval_store.sha256_text(chunk.text),
        )
        for chunk in chunks
    ]
    text_path = root / f"{document.content_hash}.txt"
    metadata_path = root / f"{document.content_hash}.json"
    archive_path: Path | None = None
    if document.archive_bytes:
        if hashlib.sha256(document.archive_bytes).hexdigest() != document.archive_hash:
            raise SourceFetchError("archive_hash_mismatch", "Fetched archive hash does not match its bytes")
        archive_root = root / "archives"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_path = archive_root / f"{document.archive_hash}.bin"
    metadata = {
        "schema_version": "corpus_staging/v1",
        "stage_id": stage_id,
        "document_id": document_id,
        "chunk_ids": chunk_ids,
        "expected_point_count": len(chunk_ids),
        "source_id": rule.source_id,
        "source_name": rule.source_name,
        "canonical_url": document.canonical_url,
        "source_url": document.source_url,
        "title": document.title,
        "fetched_at": document.fetched_at,
        "published_at": document.published_at,
        "content_type": document.content_type,
        "content_hash": document.content_hash,
        "archive_hash": document.archive_hash,
        "archive_path": (
            str(Path("archives") / archive_path.name) if archive_path is not None else None
        ),
        "extraction_version": document.extraction_version,
        "etag": document.etag,
        "last_modified": document.last_modified,
        "rights_basis_id": rule.license_id,
        "rights_basis_url": rule.license_url,
        "rights_status": "pending_document_review" if rule.rights_review_required else "source_approved",
        "retention_policy": rule.retention_policy,
        "retention_days": rule.retention_days,
    }
    encoded_text = document.text.encode("utf-8")
    encoded_metadata = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _write_content_addressed(text_path, encoded_text)
    if archive_path is not None:
        _write_content_addressed(archive_path, document.archive_bytes)
    _write_content_addressed(metadata_path, encoded_metadata)
    return StagedSourceRecord(
        schema_version="corpus_staging/v1",
        stage_id=stage_id,
        source_id=rule.source_id,
        canonical_url=document.canonical_url,
        content_hash=document.content_hash,
        archive_hash=document.archive_hash,
        extraction_version=document.extraction_version,
        metadata_path=str(metadata_path),
        text_path=str(text_path),
        rights_status=str(metadata["rights_status"]),
    )


def parse_content_type(header: str | None) -> tuple[str, str | None]:
    mime, _, parameters = str(header or "").partition(";")
    charset = None
    for parameter in parameters.split(";"):
        key, separator, value = parameter.partition("=")
        if separator and key.strip().casefold() == "charset":
            charset = value.strip().strip('"') or None
    return mime.strip().casefold(), charset


def decode_bytes(body: bytes, charset: str | None) -> str:
    for encoding in tuple(filter(None, (charset, "utf-8", "utf-8-sig", "latin-1"))):
        try:
            return body.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


def _require_public_https(
    url: str,
    *,
    resolver: Callable[..., Any],
    deadline: float | None = None,
) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise SourceFetchError("unsafe_url", "Source URL must use HTTPS")
    try:
        if deadline is None:
            addresses = resolver(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        else:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="source-dns")
            future = executor.submit(
                resolver,
                parsed.hostname,
                parsed.port or 443,
                type=socket.SOCK_STREAM,
            )
            try:
                addresses = future.result(timeout=_remaining(deadline))
            except FutureTimeoutError as exc:
                future.cancel()
                raise SourceFetchError("timeout", "Source DNS resolution exceeded the deadline") from exc
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
    except OSError as exc:
        raise SourceFetchError("dns_failure", "Could not resolve source host") from exc
    if not addresses:
        raise SourceFetchError("dns_failure", "Source host returned no addresses")
    for record in addresses:
        try:
            address = ipaddress.ip_address(record[4][0])
        except (IndexError, TypeError, ValueError) as exc:
            raise SourceFetchError("dns_failure", "Source host returned an invalid address") from exc
        if not address.is_global:
            raise SourceFetchError("unsafe_address", "Source host resolved to a non-public address")


def _remaining(deadline: float) -> float:
    remaining = float(deadline) - time.monotonic()
    if remaining <= 0:
        raise SourceFetchError("timeout", "Source fetch deadline expired")
    return max(0.001, remaining)


def _clean_header(value: Any) -> str | None:
    cleaned = str(value or "").strip()
    return cleaned[:1000] or None


def _first_date(values: list[str]) -> str | None:
    for value in values:
        try:
            return retrieval_store.normalise_datetime(value)
        except (TypeError, ValueError):
            continue
    return None


def _write_content_addressed(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise SourceFetchError("staging_collision", "Content-addressed staging collision")
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
