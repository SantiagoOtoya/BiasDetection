#!/usr/bin/env python
"""Optional, policy-agnostic Qdrant storage for trusted corpus chunks.

This module deliberately does not select an evidence retriever or issue vector
searches.  It provides the durable corpus side of that future work while
remaining importable when ``qdrant-client`` is not installed.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.parse import quote


CORPUS_SCHEMA_VERSION = "trusted_corpus/v1"
CHUNKING_VERSION = "word_window/v1"
DEFAULT_COLLECTION = "trusted_corpus_v1"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_VECTOR_NAME = "content"
DEFAULT_CONTENT_FIELD = "chunk_text"
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_CHUNK_WORDS = 350
DEFAULT_CHUNK_OVERLAP_WORDS = 50
POINT_NAMESPACE = uuid.UUID("0a0e6d4c-1ef2-5a29-b33b-66ea281d2dc2")
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")


class CorpusStoreError(RuntimeError):
    """Base class for expected corpus-store failures."""


class CorpusConfigurationError(CorpusStoreError):
    """Raised for invalid optional Qdrant configuration."""


class CollectionSchemaError(CorpusStoreError):
    """Raised when an existing collection cannot safely be reused."""


class QdrantUnavailableError(CorpusStoreError):
    """Raised internally when the optional client or server is unavailable."""


@dataclass(frozen=True)
class QdrantCorpusConfig:
    """Configuration for either embedded-local or remote Qdrant mode."""

    local_path: str | None = None
    url: str | None = None
    api_key: str | None = None
    collection: str = DEFAULT_COLLECTION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    vector_name: str = DEFAULT_VECTOR_NAME
    content_field: str = DEFAULT_CONTENT_FIELD
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        local_path = self.local_path.strip() if self.local_path else None
        url = self.url.strip() if self.url else None
        api_key = self.api_key.strip() if self.api_key else None
        object.__setattr__(self, "local_path", local_path or None)
        object.__setattr__(self, "url", url or None)
        object.__setattr__(self, "api_key", api_key or None)

        if local_path and url:
            raise CorpusConfigurationError(
                "QDRANT_LOCAL_PATH/QDRANT_PATH and QDRANT_URL cannot both be set"
            )
        if local_path and api_key:
            raise CorpusConfigurationError("QDRANT_API_KEY is only valid with QDRANT_URL")
        if url:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise CorpusConfigurationError("QDRANT_URL must be an absolute http(s) URL")
        for field_name in ("collection", "vector_name", "content_field"):
            value = str(getattr(self, field_name) or "").strip()
            if not value or not COLLECTION_NAME_RE.fullmatch(value):
                raise CorpusConfigurationError(
                    f"{field_name} must contain only letters, numbers, '_', '-', or '.'"
                )
            object.__setattr__(self, field_name, value)
        embedding_model = str(self.embedding_model or "").strip()
        if not embedding_model:
            raise CorpusConfigurationError("QDRANT_EMBEDDING_MODEL must not be empty")
        object.__setattr__(self, "embedding_model", embedding_model)
        if float(self.timeout_seconds) <= 0:
            raise CorpusConfigurationError("QDRANT timeout_seconds must be greater than zero")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

    @property
    def mode(self) -> str:
        if self.local_path:
            return "local"
        if self.url:
            return "remote"
        return "disabled"

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "QdrantCorpusConfig":
        env = environment if environment is not None else os.environ
        local_path = (env.get("QDRANT_LOCAL_PATH") or "").strip() or None
        legacy_path = (env.get("QDRANT_PATH") or "").strip() or None
        if local_path and legacy_path and local_path != legacy_path:
            raise CorpusConfigurationError(
                "QDRANT_LOCAL_PATH and QDRANT_PATH disagree; set only one path"
            )
        return cls(
            local_path=local_path or legacy_path,
            url=(env.get("QDRANT_URL") or "").strip() or None,
            api_key=(env.get("QDRANT_API_KEY") or "").strip() or None,
            collection=(env.get("QDRANT_COLLECTION") or DEFAULT_COLLECTION),
            embedding_model=(
                env.get("QDRANT_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
            ),
            vector_name=(env.get("QDRANT_VECTOR_NAME") or DEFAULT_VECTOR_NAME),
            content_field=(env.get("QDRANT_CONTENT_FIELD") or DEFAULT_CONTENT_FIELD),
            timeout_seconds=float(
                env.get("QDRANT_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS
            ),
        )


@dataclass(frozen=True)
class PayloadIndexSpec:
    field_name: str
    schema_type: str


@dataclass(frozen=True)
class CorpusCollectionSchema:
    vector_size: int
    vector_name: str
    content_field: str
    embedding_model: str
    schema_version: str = CORPUS_SCHEMA_VERSION
    chunking_version: str = CHUNKING_VERSION

    def __post_init__(self) -> None:
        if int(self.vector_size) <= 0:
            raise ValueError("vector_size must be greater than zero")
        object.__setattr__(self, "vector_size", int(self.vector_size))

    @property
    def collection_metadata(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "embedding_model": self.embedding_model,
            "vector_name": self.vector_name,
            "vector_distance": "cosine",
            "content_field": self.content_field,
            "chunking_version": self.chunking_version,
        }

    @property
    def payload_indexes(self) -> tuple[PayloadIndexSpec, ...]:
        return (
            PayloadIndexSpec("document_id", "keyword"),
            PayloadIndexSpec("chunk_id", "keyword"),
            PayloadIndexSpec("content_hash", "keyword"),
            PayloadIndexSpec("canonical_url", "keyword"),
            PayloadIndexSpec("source_id", "keyword"),
            PayloadIndexSpec("source_domain", "keyword"),
            PayloadIndexSpec("source_tier", "keyword"),
            PayloadIndexSpec("license_id", "keyword"),
            PayloadIndexSpec("retention_policy", "keyword"),
            PayloadIndexSpec("published_at", "datetime"),
            PayloadIndexSpec("fetched_at", "datetime"),
            PayloadIndexSpec("retention_until", "datetime"),
            PayloadIndexSpec("ingest_permitted", "bool"),
            PayloadIndexSpec("redistribution_allowed", "bool"),
        )

    def validation_error(self, collection_info: Any) -> str | None:
        vectors = _nested_value(collection_info, "config", "params", "vectors")
        if vectors is None:
            vectors = _value(collection_info, "vectors")
        if not isinstance(vectors, Mapping) or self.vector_name not in vectors:
            return f"missing named vector '{self.vector_name}'"
        vector = vectors[self.vector_name]
        size = _value(vector, "size")
        if size is None or int(size) != self.vector_size:
            return f"vector '{self.vector_name}' has size {size!r}, expected {self.vector_size}"
        distance = _normalise_distance(_value(vector, "distance"))
        if distance != "cosine":
            return f"vector '{self.vector_name}' uses {distance or 'no'} distance, expected cosine"

        metadata = _nested_value(collection_info, "config", "metadata")
        if metadata is None:
            metadata = _value(collection_info, "metadata")
        if not isinstance(metadata, Mapping):
            return "collection metadata is missing"
        for key, expected in self.collection_metadata.items():
            if str(metadata.get(key) or "") != expected:
                return (
                    f"collection metadata {key!r} is {metadata.get(key)!r}, "
                    f"expected {expected!r}"
                )
        return None


@dataclass(frozen=True)
class SourceMetadata:
    source_id: str
    source_name: str
    source_url: str
    canonical_url: str
    source_domain: str
    source_tier: str
    source_policy_version: str
    license_id: str
    license_url: str
    redistribution_allowed: bool
    retention_policy: str
    retention_days: int
    ingest_permitted: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "source_name",
            "source_url",
            "canonical_url",
            "source_domain",
            "source_tier",
            "source_policy_version",
            "license_id",
            "license_url",
            "retention_policy",
        ):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value)
        if int(self.retention_days) <= 0:
            raise ValueError("retention_days must be greater than zero")
        object.__setattr__(self, "retention_days", int(self.retention_days))


@dataclass(frozen=True)
class CorpusDocument:
    title: str
    text: str
    source: SourceMetadata
    published_at: str | None = None
    fetched_at: str | None = None


@dataclass(frozen=True)
class CorpusChunk:
    index: int
    start_word: int
    end_word: int
    text: str


@dataclass(frozen=True)
class CorpusPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class CorpusPointBatch:
    document_id: str
    content_hash: str
    points: tuple[CorpusPoint, ...]


@dataclass(frozen=True)
class StoredCorpusPoint:
    id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ScoredCorpusPoint:
    """A payload-bearing dense-query result from the configured corpus."""

    id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class MetadataFieldFilter:
    field_name: str
    any_of: tuple[str, ...] = ()
    gte: str | None = None
    lte: str | None = None


@dataclass(frozen=True)
class MetadataFilterSpec:
    conditions: tuple[MetadataFieldFilter, ...] = ()


@dataclass(frozen=True)
class CorpusMetadataFilter:
    source_tiers: tuple[str, ...] = ()
    source_domains: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    content_hashes: tuple[str, ...] = ()
    license_ids: tuple[str, ...] = ()
    published_from: str | None = None
    published_to: str | None = None
    fetched_from: str | None = None
    fetched_to: str | None = None
    retained_after: str | None = None


@dataclass(frozen=True)
class StoreHealth:
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class CorpusIngestionResult:
    status: str
    collection: str
    document_id: str | None = None
    content_hash: str | None = None
    point_count: int = 0
    duplicate_document_id: str | None = None
    reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "collection": self.collection,
            "document_id": self.document_id,
            "content_hash": self.content_hash,
            "point_count": self.point_count,
            "duplicate_document_id": self.duplicate_document_id,
            "reason": self.reason,
        }


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int:
        """Return the fixed output dimension."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode one dense vector for each text."""


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def canonicalize_document_url(url: str) -> str | None:
    """Canonicalize a permitted HTTPS URL for stable corpus identity."""

    try:
        parsed = urlsplit(str(url).strip())
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
    netloc = hostname if not port or port == 443 else f"{hostname}:{port}"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in TRACKING_QUERY_KEYS
        and not key.casefold().startswith("utm_")
    ]
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunsplit(("https", netloc, parsed.path or "/", query, ""))


def normalise_domain(url_or_domain: str) -> str:
    value = str(url_or_domain or "").strip().casefold()
    parsed = urlsplit(value if "://" in value else f"https://{value}")
    domain = parsed.hostname or parsed.netloc or parsed.path
    return domain.removeprefix("www.").rstrip(".")


def normalise_datetime(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        if not raw:
            return None
        if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
            raw = f"{raw}T00:00:00+00:00"
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_document_id(canonical_url: str) -> str:
    normalized = canonicalize_document_url(canonical_url)
    if normalized is None:
        raise ValueError("A document requires a canonical HTTPS URL")
    return f"doc_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def stable_chunk_id(
    document_id: str,
    chunk_index: int,
    chunk_text_hash: str,
    *,
    chunking_version: str = CHUNKING_VERSION,
) -> str:
    identity = "\0".join(
        (document_id, str(int(chunk_index)), chunk_text_hash, chunking_version)
    )
    return f"chunk_{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def stable_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def chunk_document_text(
    text: str,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> tuple[CorpusChunk, ...]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero")
    if chunk_overlap_words < 0 or chunk_overlap_words >= chunk_words:
        raise ValueError("chunk_overlap_words must be non-negative and below chunk_words")
    words = normalise_text(text).split()
    if not words:
        raise ValueError("Cannot ingest an empty document")
    stride = chunk_words - chunk_overlap_words
    chunks: list[CorpusChunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(
            CorpusChunk(
                index=len(chunks),
                start_word=start,
                end_word=end,
                text=" ".join(words[start:end]),
            )
        )
        if end == len(words):
            break
        start += stride
    return tuple(chunks)


def build_corpus_points(
    document: CorpusDocument,
    vectors: Sequence[Sequence[float]],
    *,
    content_field: str = DEFAULT_CONTENT_FIELD,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> CorpusPointBatch:
    if not document.source.ingest_permitted:
        raise ValueError("Source metadata does not permit ingestion")
    canonical_url = canonicalize_document_url(document.source.canonical_url)
    source_url = canonicalize_document_url(document.source.source_url)
    if canonical_url is None or source_url is None:
        raise ValueError("Source metadata must contain permitted HTTPS URLs")
    chunks = chunk_document_text(
        document.text,
        chunk_words=chunk_words,
        chunk_overlap_words=chunk_overlap_words,
    )
    if len(chunks) != len(vectors):
        raise ValueError("The number of vectors must equal the number of chunks")

    document_id = stable_document_id(canonical_url)
    content_hash = sha256_text(document.text)
    fetched_at = normalise_datetime(document.fetched_at) or utc_now()
    published_at = normalise_datetime(document.published_at)
    fetched_at_dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    retention_until = (fetched_at_dt + timedelta(days=document.source.retention_days)).isoformat()
    retention_until = retention_until.replace("+00:00", "Z")
    title = normalise_text(document.title) or canonical_url

    points: list[CorpusPoint] = []
    for chunk, raw_vector in zip(chunks, vectors, strict=True):
        vector = [float(value) for value in raw_vector]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("Embeddings must be finite, non-empty numeric vectors")
        chunk_text_hash = sha256_text(chunk.text)
        chunk_id = stable_chunk_id(document_id, chunk.index, chunk_text_hash)
        point_id = stable_point_id(chunk_id)
        points.append(
            CorpusPoint(
                id=point_id,
                vector=vector,
                payload={
                    "schema_version": CORPUS_SCHEMA_VERSION,
                    "document_id": document_id,
                    "point_id": point_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk.index,
                    "chunk_start_word": chunk.start_word,
                    "chunk_end_word": chunk.end_word,
                    content_field: chunk.text,
                    "chunk_text_hash": chunk_text_hash,
                    "content_hash": content_hash,
                    "source_url": source_url,
                    "canonical_url": canonical_url,
                    "title": title,
                    "source_id": document.source.source_id,
                    "source_name": document.source.source_name,
                    "source_domain": document.source.source_domain,
                    "source_tier": document.source.source_tier,
                    "source_policy_version": document.source.source_policy_version,
                    "published_at": published_at,
                    "fetched_at": fetched_at,
                    "license_id": document.source.license_id,
                    "license_url": document.source.license_url,
                    "redistribution_allowed": bool(
                        document.source.redistribution_allowed
                    ),
                    "retention_policy": document.source.retention_policy,
                    "retention_until": retention_until,
                    "ingest_permitted": bool(document.source.ingest_permitted),
                },
            )
        )
    return CorpusPointBatch(
        document_id=document_id,
        content_hash=content_hash,
        points=tuple(points),
    )


def build_metadata_filter(criteria: CorpusMetadataFilter) -> MetadataFilterSpec:
    conditions: list[MetadataFieldFilter] = []
    for field_name, values in (
        ("source_tier", criteria.source_tiers),
        ("source_domain", criteria.source_domains),
        ("source_id", criteria.source_ids),
        ("document_id", criteria.document_ids),
        ("content_hash", criteria.content_hashes),
        ("license_id", criteria.license_ids),
    ):
        normalized = tuple(sorted({str(value).strip() for value in values if str(value).strip()}))
        if normalized:
            conditions.append(MetadataFieldFilter(field_name=field_name, any_of=normalized))
    for field_name, lower, upper in (
        ("published_at", criteria.published_from, criteria.published_to),
        ("fetched_at", criteria.fetched_from, criteria.fetched_to),
        ("retention_until", criteria.retained_after, None),
    ):
        gte = normalise_datetime(lower) if lower else None
        lte = normalise_datetime(upper) if upper else None
        if gte or lte:
            conditions.append(MetadataFieldFilter(field_name=field_name, gte=gte, lte=lte))
    return MetadataFilterSpec(tuple(conditions))


class QdrantCorpusStore:
    """A lazily connected Qdrant store with safe unavailable-state results."""

    def __init__(
        self,
        config: QdrantCorpusConfig,
        *,
        client: Any | None = None,
        client_factory: Callable[[QdrantCorpusConfig], Any] | None = None,
        models_module: Any | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._client_factory = client_factory
        self._models = models_module

    def health(self) -> StoreHealth:
        if self.config.mode == "disabled":
            return StoreHealth("disabled", "Qdrant is not configured")
        try:
            client = self._get_client()
            get_collections = getattr(client, "get_collections", None)
            if callable(get_collections):
                get_collections()
        except QdrantUnavailableError as exc:
            return StoreHealth("unavailable", str(exc))
        except Exception:
            return StoreHealth("unavailable", "Qdrant health check failed")
        return StoreHealth("ready")

    def list_collection_names(self) -> tuple[str, ...]:
        """Return exact collection names without exposing connection settings."""

        client = self._get_client()
        try:
            response = client.get_collections()
            collections = _value(response, "collections", ()) or ()
            names = {
                str(_value(collection, "name", "") or "").strip()
                for collection in collections
            }
            return tuple(sorted(name for name in names if name))
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant collection listing failed") from exc

    def ensure_collection(self, schema: CorpusCollectionSchema) -> bool:
        """Create or validate the collection and its filter indexes.

        Returns ``True`` only when the collection was created during this call.
        """

        client = self._get_client()
        try:
            exists = bool(client.collection_exists(self.config.collection))
            created = False
            if not exists:
                create_options: dict[str, Any] = {}
                if self._models is not None and hasattr(self._models, "StrictModeConfig"):
                    create_options["strict_mode_config"] = self._models.StrictModeConfig(
                        enabled=True,
                        unindexed_filtering_retrieve=False,
                        unindexed_filtering_update=False,
                    )
                client.create_collection(
                    collection_name=self.config.collection,
                    vectors_config=self._vector_config(schema),
                    metadata=schema.collection_metadata,
                    **create_options,
                )
                created = True
            info = client.get_collection(self.config.collection)
            mismatch = schema.validation_error(info)
            if mismatch:
                raise CollectionSchemaError(
                    f"Collection '{self.config.collection}' is incompatible: {mismatch}"
                )
            self._ensure_payload_indexes(client, schema)
            if self.config.mode == "remote":
                refreshed = client.get_collection(self.config.collection)
                strict_error = self.production_preflight_error(refreshed, schema)
                if strict_error:
                    raise CollectionSchemaError(
                        f"Collection '{self.config.collection}' failed production preflight: {strict_error}"
                    )
            return created
        except CollectionSchemaError:
            raise
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant collection operations failed") from exc

    def production_preflight_error(
        self,
        collection_info: Any,
        schema: CorpusCollectionSchema,
    ) -> str | None:
        mismatch = schema.validation_error(collection_info)
        if mismatch:
            return mismatch
        payload_schema = _value(collection_info, "payload_schema", {}) or {}
        if not isinstance(payload_schema, Mapping):
            return "payload schema is unavailable"
        missing = sorted(
            index.field_name for index in schema.payload_indexes
            if index.field_name not in payload_schema
        )
        if missing:
            return "missing payload indexes: " + ", ".join(missing)
        strict = _nested_value(collection_info, "config", "strict_mode_config")
        if strict is None or not bool(_value(strict, "enabled", False)):
            return "strict mode is not enabled"
        if _value(strict, "unindexed_filtering_retrieve", None) is not False:
            return "strict mode permits unindexed retrieval filtering"
        if _value(strict, "unindexed_filtering_update", None) is not False:
            return "strict mode permits unindexed update filtering"
        return None

    def list_points(
        self,
        criteria: CorpusMetadataFilter | None = None,
        *,
        limit: int = 100,
    ) -> tuple[StoredCorpusPoint, ...]:
        if limit <= 0:
            return ()
        client = self._get_client()
        filter_value = self._qdrant_filter(build_metadata_filter(criteria or CorpusMetadataFilter()))
        try:
            try:
                response = client.scroll(
                    collection_name=self.config.collection,
                    scroll_filter=filter_value,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
            except TypeError:
                response = client.scroll(
                    collection_name=self.config.collection,
                    filter=filter_value,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant metadata filtering failed") from exc
        records = response[0] if isinstance(response, tuple) else _value(response, "points", response)
        return tuple(_stored_point(record) for record in records)

    def list_all_points(
        self,
        criteria: CorpusMetadataFilter | None = None,
        *,
        page_size: int = 256,
        max_points: int | None = None,
    ) -> tuple[StoredCorpusPoint, ...]:
        """Scroll every matching point exactly, following opaque server offsets."""

        if page_size <= 0 or (max_points is not None and max_points <= 0):
            return ()
        client = self._get_client()
        try:
            if not bool(client.collection_exists(self.config.collection)):
                return ()
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant exact metadata scroll failed") from exc
        filter_value = self._qdrant_filter(build_metadata_filter(criteria or CorpusMetadataFilter()))
        records: list[StoredCorpusPoint] = []
        offset: Any = None
        while True:
            limit = page_size
            if max_points is not None:
                limit = min(limit, max_points - len(records))
                if limit <= 0:
                    break
            kwargs = {
                "collection_name": self.config.collection,
                "scroll_filter": filter_value,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False,
            }
            if offset is not None:
                kwargs["offset"] = offset
            try:
                response = client.scroll(**kwargs)
            except TypeError:
                # Compatibility with older clients and simple test fakes.
                if "offset" in kwargs:
                    raise QdrantUnavailableError(
                        "The configured Qdrant client cannot perform exact paginated scrolls"
                    )
                try:
                    response = client.scroll(**kwargs)
                except TypeError:
                    kwargs["filter"] = kwargs.pop("scroll_filter")
                    response = client.scroll(**kwargs)
                offset = None
            except Exception as exc:
                raise QdrantUnavailableError("Qdrant exact metadata scroll failed") from exc
            page = response[0] if isinstance(response, tuple) else _value(response, "points", response)
            next_offset = response[1] if isinstance(response, tuple) and len(response) > 1 else _value(
                response, "next_page_offset"
            )
            page_records = [_stored_point(record) for record in (page or ())]
            records.extend(page_records)
            if not page_records or next_offset is None or next_offset == offset:
                break
            offset = next_offset
        return tuple(records)

    def collection_info(self) -> Any | None:
        client = self._get_client()
        try:
            if not bool(client.collection_exists(self.config.collection)):
                return None
            return client.get_collection(self.config.collection)
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant collection inspection failed") from exc

    def server_version(self) -> str | None:
        client = self._get_client()
        try:
            info = client.info()
            value = str(_value(info, "version", "") or "").strip()
            return value or None
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant server version inspection failed") from exc

    def delete_documents(
        self,
        document_ids: Sequence[str],
        *,
        batch_size: int = 128,
    ) -> int:
        """Delete all points for reviewed document IDs in bounded batches."""

        normalized = tuple(sorted({str(value).strip() for value in document_ids if str(value).strip()}))
        if not normalized:
            return 0
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        points = self.list_all_points(CorpusMetadataFilter(document_ids=normalized))
        point_ids = sorted({point.id for point in points})
        for start in range(0, len(point_ids), batch_size):
            self._delete_point_ids(set(point_ids[start : start + batch_size]))
        return len(point_ids)

    def create_snapshot(self) -> str:
        client = self._get_client()
        try:
            response = client.create_snapshot(
                collection_name=self.config.collection,
                wait=True,
            )
            name = str(_value(response, "name", "") or "").strip()
            if not name:
                raise ValueError("snapshot response omitted its name")
            return name
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant snapshot creation failed") from exc

    def list_snapshots(self) -> tuple[Any, ...]:
        client = self._get_client()
        try:
            return tuple(client.list_snapshots(collection_name=self.config.collection))
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant snapshot listing failed") from exc

    def delete_snapshot(self, snapshot_name: str) -> None:
        client = self._get_client()
        try:
            client.delete_snapshot(
                collection_name=self.config.collection,
                snapshot_name=snapshot_name,
                wait=True,
            )
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant snapshot deletion failed") from exc

    def download_snapshot(self, snapshot_name: str, destination: Path | str) -> Path:
        client = self._get_client()
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            download = getattr(client, "download_snapshot", None)
            if callable(download):
                result = download(
                    collection_name=self.config.collection,
                    snapshot_name=snapshot_name,
                    location=str(target),
                )
                actual = Path(result) if isinstance(result, (str, Path)) else target
                if actual != target and actual.is_file():
                    actual.replace(target)
            else:
                if self.config.mode != "remote" or not self.config.url or not self.config.api_key:
                    raise ValueError("Snapshot downloads require authenticated Qdrant Cloud")
                import httpx

                url = (
                    self.config.url.rstrip("/")
                    + f"/collections/{quote(self.config.collection, safe='')}/snapshots/"
                    + quote(snapshot_name, safe="")
                )
                with httpx.stream(
                    "GET",
                    url,
                    headers={"api-key": self.config.api_key},
                    timeout=max(30.0, self.config.timeout_seconds),
                    follow_redirects=False,
                ) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as handle:
                        for chunk in response.iter_bytes():
                            handle.write(chunk)
                temporary.replace(target)
            if not target.is_file() or target.stat().st_size <= 0:
                raise ValueError("downloaded snapshot is empty")
            return target
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant snapshot download failed") from exc
        finally:
            if temporary.exists():
                temporary.unlink()

    def restore_snapshot_to_collection(
        self,
        snapshot_path: Path | str,
        collection_name: str,
    ) -> None:
        """Upload a snapshot into an explicitly isolated collection."""

        if collection_name == self.config.collection or not COLLECTION_NAME_RE.fullmatch(collection_name):
            raise ValueError("Restore verification requires a valid isolated collection name")
        path = Path(snapshot_path)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError("Snapshot file is missing or empty")
        client = self._get_client()
        try:
            upload = getattr(client, "upload_snapshot", None)
            if callable(upload):
                upload(
                    collection_name=collection_name,
                    snapshot_path=str(path),
                    wait=True,
                )
            else:
                snapshots_api = client.http.snapshots_api
                with path.open("rb") as handle:
                    snapshots_api.recover_from_uploaded_snapshot(
                        collection_name=collection_name,
                        wait=True,
                        checksum=_sha256_file(path),
                        snapshot=handle,
                    )
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant snapshot restore failed") from exc

    def delete_collection(self, collection_name: str) -> None:
        if not COLLECTION_NAME_RE.fullmatch(str(collection_name or "")):
            raise ValueError("Invalid collection name")
        client = self._get_client()
        try:
            if bool(client.collection_exists(collection_name)):
                client.delete_collection(collection_name=collection_name)
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant collection deletion failed") from exc

    def search_points(
        self,
        query_vector: Sequence[float],
        *,
        criteria: CorpusMetadataFilter | None = None,
        limit: int = 10,
    ) -> tuple[ScoredCorpusPoint, ...]:
        """Query the configured named cosine vector without mutating the corpus.

        A missing collection is an empty corpus rather than an operational error.
        Configuration, schema, and server failures remain explicit so callers can
        safely distinguish unavailable corpus retrieval from no matching evidence.
        """

        if limit <= 0:
            return ()
        vector = [float(value) for value in query_vector]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("Query embeddings must be finite, non-empty numeric vectors")

        client = self._get_client()
        try:
            if not bool(client.collection_exists(self.config.collection)):
                return ()

            schema = CorpusCollectionSchema(
                vector_size=len(vector),
                vector_name=self.config.vector_name,
                content_field=self.config.content_field,
                embedding_model=self.config.embedding_model,
            )
            collection_info = client.get_collection(self.config.collection)
            mismatch = schema.validation_error(collection_info)
            if mismatch:
                raise CollectionSchemaError(
                    f"Collection '{self.config.collection}' is incompatible: {mismatch}"
                )
            if self.config.mode == "remote":
                strict_error = self.production_preflight_error(collection_info, schema)
                if strict_error:
                    raise CollectionSchemaError(
                        f"Collection '{self.config.collection}' failed production preflight: {strict_error}"
                    )

            active_criteria = criteria or CorpusMetadataFilter()
            if not active_criteria.retained_after:
                active_criteria = replace(active_criteria, retained_after=utc_now())
            filter_value = self._qdrant_filter(build_metadata_filter(active_criteria))

            query_points = getattr(client, "query_points", None)
            if callable(query_points):
                response = query_points(
                    collection_name=self.config.collection,
                    query=vector,
                    using=self.config.vector_name,
                    query_filter=filter_value,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
            else:
                search = getattr(client, "search", None)
                if not callable(search):
                    raise QdrantUnavailableError(
                        "The configured Qdrant client does not support vector queries"
                    )
                response = search(
                    collection_name=self.config.collection,
                    query_vector=(self.config.vector_name, vector),
                    query_filter=filter_value,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
        except CollectionSchemaError:
            raise
        except QdrantUnavailableError:
            raise
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant vector search failed") from exc

        records = _value(response, "points", response)
        try:
            points = [_scored_point(record) for record in records]
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant returned an invalid vector-search result") from exc
        return tuple(sorted(points, key=lambda point: (-point.score, point.id)))

    def ingest_document(
        self,
        document: CorpusDocument,
        vectors: Sequence[Sequence[float]],
        *,
        chunk_words: int = DEFAULT_CHUNK_WORDS,
        chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    ) -> CorpusIngestionResult:
        if self.config.mode == "disabled":
            return CorpusIngestionResult(
                status="disabled",
                collection=self.config.collection,
                reason="Qdrant is not configured",
            )
        try:
            batch = build_corpus_points(
                document,
                vectors,
                content_field=self.config.content_field,
                chunk_words=chunk_words,
                chunk_overlap_words=chunk_overlap_words,
            )
            schema = CorpusCollectionSchema(
                vector_size=len(batch.points[0].vector),
                vector_name=self.config.vector_name,
                content_field=self.config.content_field,
                embedding_model=self.config.embedding_model,
            )
            self.ensure_collection(schema)
            duplicate = self.list_points(
                CorpusMetadataFilter(content_hashes=(batch.content_hash,)), limit=1
            )
            if duplicate:
                duplicate_document_id = str(
                    duplicate[0].payload.get("document_id") or ""
                ) or None
                return CorpusIngestionResult(
                    status="duplicate",
                    collection=self.config.collection,
                    document_id=batch.document_id,
                    content_hash=batch.content_hash,
                    duplicate_document_id=duplicate_document_id,
                    reason=(
                        "document content is already stored"
                        if duplicate_document_id == batch.document_id
                        else "identical document content is already stored under another source"
                    ),
                )

            existing = self.list_points(
                CorpusMetadataFilter(document_ids=(batch.document_id,)), limit=10000
            )
            self._upsert(batch.points, schema)
            stale_ids = {point.id for point in existing} - {
                point.id for point in batch.points
            }
            if stale_ids:
                self._delete_point_ids(stale_ids)
            return CorpusIngestionResult(
                status="replaced" if existing else "ingested",
                collection=self.config.collection,
                document_id=batch.document_id,
                content_hash=batch.content_hash,
                point_count=len(batch.points),
            )
        except CollectionSchemaError as exc:
            return CorpusIngestionResult(
                status="invalid_schema",
                collection=self.config.collection,
                reason=str(exc),
            )
        except QdrantUnavailableError as exc:
            return CorpusIngestionResult(
                status="unavailable",
                collection=self.config.collection,
                reason=str(exc),
            )

    def _get_client(self) -> Any:
        if self.config.mode == "disabled":
            raise QdrantUnavailableError("Qdrant is not configured")
        if self._client is not None:
            return self._client
        try:
            if self._client_factory is not None:
                self._client = self._client_factory(self.config)
                return self._client
            from qdrant_client import QdrantClient, models  # type: ignore[import-not-found]

            self._models = models
            if self.config.mode == "local":
                self._client = QdrantClient(path=self.config.local_path)
            else:
                self._client = QdrantClient(
                    url=self.config.url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout_seconds,
                )
            return self._client
        except ImportError as exc:
            raise QdrantUnavailableError(
                "qdrant-client is not installed; install requirements-qdrant.txt"
            ) from exc
        except Exception as exc:
            raise QdrantUnavailableError("Could not initialize the configured Qdrant client") from exc

    def _vector_config(self, schema: CorpusCollectionSchema) -> Any:
        if self._models is None:
            return {
                schema.vector_name: {"size": schema.vector_size, "distance": "Cosine"}
            }
        return {
            schema.vector_name: self._models.VectorParams(
                size=schema.vector_size,
                distance=self._models.Distance.COSINE,
            )
        }

    def _ensure_payload_indexes(self, client: Any, schema: CorpusCollectionSchema) -> None:
        for index in schema.payload_indexes:
            try:
                client.create_payload_index(
                    collection_name=self.config.collection,
                    field_name=index.field_name,
                    field_schema=self._payload_schema_type(index.schema_type),
                    wait=True,
                )
            except NotImplementedError:
                if self.config.mode == "local":
                    continue
                raise
            except Exception as exc:
                if "already exists" in str(exc).casefold():
                    continue
                raise

    def _payload_schema_type(self, schema_type: str) -> Any:
        if self._models is None:
            return schema_type
        mapping = {
            "keyword": self._models.PayloadSchemaType.KEYWORD,
            "datetime": self._models.PayloadSchemaType.DATETIME,
            "bool": self._models.PayloadSchemaType.BOOL,
        }
        return mapping[schema_type]

    def _qdrant_filter(self, filter_spec: MetadataFilterSpec) -> Any:
        if not filter_spec.conditions:
            return None
        if self._models is None:
            return filter_spec
        conditions: list[Any] = []
        for condition in filter_spec.conditions:
            if condition.any_of:
                conditions.append(
                    self._models.FieldCondition(
                        key=condition.field_name,
                        match=self._models.MatchAny(any=list(condition.any_of)),
                    )
                )
            if condition.gte or condition.lte:
                conditions.append(
                    self._models.FieldCondition(
                        key=condition.field_name,
                        range=self._models.DatetimeRange(
                            gte=condition.gte,
                            lte=condition.lte,
                        ),
                    )
                )
        return self._models.Filter(must=conditions)

    def _upsert(
        self, points: Sequence[CorpusPoint], schema: CorpusCollectionSchema
    ) -> None:
        client = self._get_client()
        qdrant_points: Sequence[Any]
        if self._models is None:
            qdrant_points = points
        else:
            qdrant_points = [
                self._models.PointStruct(
                    id=point.id,
                    vector={schema.vector_name: point.vector},
                    payload=point.payload,
                )
                for point in points
            ]
        try:
            client.upsert(
                collection_name=self.config.collection,
                points=qdrant_points,
                wait=True,
            )
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant upsert failed") from exc

    def _delete_point_ids(self, point_ids: set[str]) -> None:
        if not point_ids:
            return
        client = self._get_client()
        try:
            if self._models is None:
                selector: Any = tuple(sorted(point_ids))
            else:
                selector = self._models.PointIdsList(points=sorted(point_ids))
            client.delete(
                collection_name=self.config.collection,
                points_selector=selector,
                wait=True,
            )
        except Exception as exc:
            raise QdrantUnavailableError("Qdrant stale-point cleanup failed") from exc


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _nested_value(value: Any, *names: str) -> Any:
    current = value
    for name in names:
        current = _value(current, name)
        if current is None:
            return None
    return current


def _normalise_distance(value: Any) -> str:
    raw = getattr(value, "value", value)
    if raw is None:
        return ""
    return str(raw).rsplit(".", 1)[-1].casefold()


def _stored_point(record: Any) -> StoredCorpusPoint:
    if isinstance(record, StoredCorpusPoint):
        return record
    if isinstance(record, CorpusPoint):
        return StoredCorpusPoint(id=record.id, payload=dict(record.payload))
    payload = _value(record, "payload", {}) or {}
    return StoredCorpusPoint(id=str(_value(record, "id")), payload=dict(payload))


def _scored_point(record: Any) -> ScoredCorpusPoint:
    payload = _value(record, "payload", {}) or {}
    score = float(_value(record, "score"))
    if not math.isfinite(score):
        raise ValueError("Qdrant returned a non-finite score")
    return ScoredCorpusPoint(
        id=str(_value(record, "id")),
        score=score,
        payload=dict(payload),
    )
