#!/usr/bin/env python
"""Reviewed manifest and Qdrant Cloud lifecycle operations.

All mutations are dry-run unless the caller passes ``apply=True`` (or CLI
``--apply``).  Qdrant Cloud remains authoritative; this module never provisions
an embedded/local server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

import retrieval_store
import runtime_config
import source_fetch
from trusted_source_registry import TrustedSourceRegistry


MANIFEST_SCHEMA_VERSION = "trusted_corpus_manifest/v1"
POINTER_SCHEMA_VERSION = "trusted_corpus_manifest_pointer/v1"
RUN_SCHEMA_VERSION = "corpus_run/v1"
DEFAULT_CORPUS_DIR = Path(__file__).with_name(".corpus")
DEFAULT_MANIFEST_DIR = Path(__file__).with_name("corpus_manifests")
DEFAULT_BACKUP_GENERATIONS = 4
APPROVED_DOCUMENT_STATUSES = {"approved", "active"}
CANDIDATE_DOCUMENT_STATUSES = {"candidate", "pending_fetch", "pending_review", "expired"}


class CorpusOperationError(RuntimeError):
    """A lifecycle precondition, review gate, or provider operation failed."""


@dataclass(frozen=True)
class ManifestValidation:
    valid: bool
    manifest_hash: str | None
    document_count: int
    expected_point_count: int
    errors: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestDiff:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]
    unchanged: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BackupRecord:
    schema_version: str
    backup_id: str
    created_at: str
    collection: str
    snapshot_name: str
    snapshot_path: str
    snapshot_sha256: str
    archive_path: str
    archive_sha256: str
    verified: bool
    point_count: int
    qdrant_server_version: str | None = None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def load_json_object(path: Path | str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusOperationError(f"Could not read valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CorpusOperationError(f"JSON root must be an object: {path}")
    return value


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    registry: TrustedSourceRegistry | None = None,
    require_approved: bool = False,
) -> ManifestValidation:
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")
    for field in ("manifest_id", "created_at", "policy_version", "collection", "embedding"):
        if not manifest.get(field):
            errors.append(f"manifest field {field!r} is required")
    embedding = manifest.get("embedding")
    if not isinstance(embedding, Mapping):
        errors.append("embedding must be an object")
    else:
        expected_embedding = {
            "model": retrieval_store.DEFAULT_EMBEDDING_MODEL,
            "distance": "cosine",
            "vector_name": retrieval_store.DEFAULT_VECTOR_NAME,
            "content_field": retrieval_store.DEFAULT_CONTENT_FIELD,
            "chunk_words": retrieval_store.DEFAULT_CHUNK_WORDS,
            "chunk_overlap_words": retrieval_store.DEFAULT_CHUNK_OVERLAP_WORDS,
        }
        for key, expected in expected_embedding.items():
            if embedding.get(key) != expected:
                errors.append(f"embedding.{key} must be {expected!r}")
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        errors.append("documents must be an array")
        documents = []
    ids: set[str] = set()
    urls: set[str] = set()
    expected_points = 0
    for index, entry in enumerate(documents):
        prefix = f"documents[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        for field in (
            "source_id", "canonical_document_id", "canonical_url", "title",
            "extraction_version", "rights_basis", "rights_review", "refresh_after",
            "retention_until", "status", "expected_point_count",
        ):
            value = entry.get(field)
            if field not in entry or value is None or value == "":
                errors.append(f"{prefix}.{field} is required")
        canonical_id = str(entry.get("canonical_document_id") or "")
        canonical_url = retrieval_store.canonicalize_document_url(str(entry.get("canonical_url") or ""))
        if canonical_id in ids:
            errors.append(f"{prefix}.canonical_document_id is duplicated")
        ids.add(canonical_id)
        if not canonical_url:
            errors.append(f"{prefix}.canonical_url must be canonical HTTPS")
        elif canonical_url in urls:
            errors.append(f"{prefix}.canonical_url is duplicated")
        else:
            urls.add(canonical_url)
        status = str(entry.get("status") or "")
        if status not in APPROVED_DOCUMENT_STATUSES | CANDIDATE_DOCUMENT_STATUSES:
            errors.append(f"{prefix}.status is unsupported")
        point_count = entry.get("expected_point_count")
        if not isinstance(point_count, int) or point_count < 0:
            errors.append(f"{prefix}.expected_point_count must be a non-negative integer")
        else:
            expected_points += point_count
        for date_field in ("published_at", "fetched_at", "refresh_after", "retention_until"):
            date_value = entry.get(date_field)
            if date_value is not None and _parse_datetime(date_value) is None:
                errors.append(f"{prefix}.{date_field} must be ISO-8601 or null")
        rights = entry.get("rights_review")
        rights_basis = entry.get("rights_basis")
        if not isinstance(rights, Mapping):
            errors.append(f"{prefix}.rights_review must be an object")
        if (
            not isinstance(rights_basis, Mapping)
            or not rights_basis.get("id")
            or not rights_basis.get("url")
        ):
            errors.append(f"{prefix}.rights_basis needs id and url")
        if require_approved or status in APPROVED_DOCUMENT_STATUSES:
            if status not in APPROVED_DOCUMENT_STATUSES:
                errors.append(f"{prefix}.status is not approved")
            for field in ("content_hash", "archive_hash"):
                if not _is_sha256(entry.get(field)):
                    errors.append(f"{prefix}.{field} must be a SHA-256 hash")
            if _parse_datetime(entry.get("fetched_at")) is None:
                errors.append(f"{prefix}.fetched_at is required when approved")
            if not isinstance(rights, Mapping) or rights.get("status") != "approved":
                errors.append(f"{prefix}.rights_review.status must be approved")
            elif not rights.get("reviewer") or not rights.get("reviewed_at"):
                errors.append(f"{prefix}.rights_review needs reviewer and reviewed_at")
            if rights.get("third_party_marked"):
                errors.append(f"{prefix} is marked as third-party material")
            if not isinstance(point_count, int) or point_count <= 0:
                errors.append(f"{prefix}.expected_point_count must be positive when approved")
        if registry is not None and canonical_url:
            decision = registry.evaluate_url(canonical_url, purpose="ingest")
            if not decision.allowed or decision.rule is None:
                errors.append(f"{prefix} is rejected by the current source policy")
            elif entry.get("source_id") != decision.rule.source_id:
                errors.append(f"{prefix}.source_id does not match the source registry")
            elif isinstance(rights_basis, Mapping) and (
                rights_basis.get("id") != decision.rule.license_id
                or rights_basis.get("url") != decision.rule.license_url
            ):
                errors.append(f"{prefix}.rights_basis does not match the source registry")
    actual_hash = manifest_hash(manifest)
    declared_hash = manifest.get("manifest_hash")
    if declared_hash is not None and declared_hash != actual_hash:
        errors.append("manifest_hash does not match canonical manifest content")
    return ManifestValidation(
        valid=not errors,
        manifest_hash=actual_hash if not errors else None,
        document_count=len(documents),
        expected_point_count=expected_points,
        errors=tuple(errors),
    )


def approve_manifest(
    candidate_path: Path | str,
    *,
    registry: TrustedSourceRegistry,
    reviewer: str,
    manifest_dir: Path | str = DEFAULT_MANIFEST_DIR,
    apply: bool = False,
) -> dict[str, Any]:
    candidate = load_json_object(candidate_path)
    validation = validate_manifest(candidate, registry=registry, require_approved=True)
    if not validation.valid or validation.manifest_hash is None:
        raise CorpusOperationError("Manifest approval failed: " + "; ".join(validation.errors))
    reviewer_value = str(reviewer or "").strip()
    if not reviewer_value:
        raise CorpusOperationError("Manifest reviewer is required")
    approved = dict(candidate)
    approved["review"] = {
        "status": "approved",
        "reviewer": reviewer_value,
        "reviewed_at": retrieval_store.utc_now(),
    }
    approved["manifest_hash"] = manifest_hash(approved)
    root = Path(manifest_dir)
    immutable_path = root / f"manifest-{approved['manifest_hash']}.json"
    pointer_path = root / "current.json"
    result = {
        "status": "would_approve" if not apply else "approved",
        "manifest_hash": approved["manifest_hash"],
        "manifest_path": str(immutable_path),
        "pointer_path": str(pointer_path),
    }
    if not apply:
        return result
    root.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(approved, sort_keys=True, indent=2) + "\n"
    if immutable_path.exists() and immutable_path.read_text(encoding="utf-8") != encoded:
        raise CorpusOperationError("Immutable manifest path already contains different bytes")
    if not immutable_path.exists():
        _atomic_json_write(immutable_path, approved)
    pointer = {
        "schema_version": POINTER_SCHEMA_VERSION,
        "manifest_hash": approved["manifest_hash"],
        "manifest_file": immutable_path.name,
        "updated_at": retrieval_store.utc_now(),
    }
    _atomic_json_write(pointer_path, pointer)
    return result


def load_current_manifest(
    manifest_dir: Path | str = DEFAULT_MANIFEST_DIR,
    *,
    require_approved: bool = True,
) -> dict[str, Any]:
    root = Path(manifest_dir)
    pointer = load_json_object(root / "current.json")
    if pointer.get("schema_version") != POINTER_SCHEMA_VERSION:
        raise CorpusOperationError("Current-manifest pointer has an unsupported schema")
    name = str(pointer.get("manifest_file") or "")
    if Path(name).name != name:
        raise CorpusOperationError("Current-manifest pointer contains an unsafe path")
    manifest_path = (root / name).resolve()
    if root.resolve() not in manifest_path.parents:
        raise CorpusOperationError("Current-manifest pointer escapes the manifest directory")
    manifest = load_json_object(manifest_path)
    actual = manifest_hash(manifest)
    if pointer.get("manifest_hash") != actual or manifest.get("manifest_hash") != actual:
        raise CorpusOperationError("Current-manifest pointer hash does not match the manifest")
    if require_approved and (manifest.get("review") or {}).get("status") != "approved":
        raise CorpusOperationError("Current manifest is not reviewed and approved")
    return manifest


def diff_manifests(current: Mapping[str, Any], candidate: Mapping[str, Any]) -> ManifestDiff:
    before = {
        str(entry.get("canonical_document_id")): entry
        for entry in current.get("documents", ())
        if isinstance(entry, Mapping)
    }
    after = {
        str(entry.get("canonical_document_id")): entry
        for entry in candidate.get("documents", ())
        if isinstance(entry, Mapping)
    }
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    changed = sorted(
        key for key in before.keys() & after.keys()
        if canonical_json_bytes(before[key]) != canonical_json_bytes(after[key])
    )
    unchanged = sorted((before.keys() & after.keys()) - set(changed))
    return ManifestDiff(tuple(added), tuple(removed), tuple(changed), tuple(unchanged))


def batch_ingest_manifest(
    store: retrieval_store.QdrantCorpusStore,
    manifest: Mapping[str, Any],
    registry: TrustedSourceRegistry,
    *,
    staging_dir: Path | str,
    embedding_provider: retrieval_store.EmbeddingProvider | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Idempotently ingest one reviewed immutable manifest as a batch."""

    validation = validate_manifest(manifest, registry=registry, require_approved=True)
    if not validation.valid:
        raise CorpusOperationError("Batch manifest is invalid: " + "; ".join(validation.errors))
    if (manifest.get("review") or {}).get("status") != "approved":
        raise CorpusOperationError("Batch ingestion requires a reviewed immutable manifest")
    documents = [
        entry for entry in manifest.get("documents", ())
        if isinstance(entry, Mapping) and entry.get("status") in APPROVED_DOCUMENT_STATUSES
    ]
    plan = {
        "status": "would_ingest" if not apply else "applied",
        "document_count": len(documents),
        "expected_point_count": validation.expected_point_count,
        "results": [],
    }
    if not apply:
        return plan
    provider = embedding_provider
    if provider is None:
        from ingest_trusted_corpus import SentenceTransformerEmbedder

        provider = SentenceTransformerEmbedder(store.config.embedding_model)
    for entry in documents:
        corpus_document = document_from_staging(entry, registry, staging_dir=staging_dir)
        chunks = retrieval_store.chunk_document_text(corpus_document.text)
        vectors = provider.embed([chunk.text for chunk in chunks])
        result = store.ingest_document(corpus_document, vectors)
        if result.status not in {"ingested", "replaced", "duplicate"}:
            raise CorpusOperationError(
                f"Batch ingestion failed for {entry.get('canonical_document_id')}: {result.status}"
            )
        if result.point_count and result.point_count != int(entry.get("expected_point_count") or 0):
            raise CorpusOperationError("Ingested point count differs from the reviewed manifest")
        plan["results"].append(result.to_json())
    return plan


def document_from_staging(
    entry: Mapping[str, Any],
    registry: TrustedSourceRegistry,
    *,
    staging_dir: Path | str,
) -> retrieval_store.CorpusDocument:
    content_hash = str(entry.get("content_hash") or "")
    if not _is_sha256(content_hash):
        raise CorpusOperationError("Approved entry lacks a valid content hash")
    text_path = Path(staging_dir) / f"{content_hash}.txt"
    metadata_path = Path(staging_dir) / f"{content_hash}.json"
    try:
        text = text_path.read_text(encoding="utf-8")
        staged = load_json_object(metadata_path)
    except OSError as exc:
        raise CorpusOperationError("Reviewed staged content is missing") from exc
    if retrieval_store.sha256_text(text) != content_hash:
        raise CorpusOperationError("Staged text hash does not match the manifest")
    if staged.get("archive_hash") != entry.get("archive_hash"):
        raise CorpusOperationError("Staged archive hash does not match the manifest")
    rights = entry.get("rights_review") or {}
    rule = registry.require_document_rights(
        str(entry.get("canonical_url") or ""),
        reviewer=str(rights.get("reviewer") or ""),
        reviewed_at=str(rights.get("reviewed_at") or ""),
        third_party_marked=bool(rights.get("third_party_marked")),
    )
    canonical_url = str(entry["canonical_url"])
    expected_document_id = retrieval_store.stable_document_id(canonical_url)
    if entry.get("canonical_document_id") != expected_document_id:
        raise CorpusOperationError("Manifest canonical document ID is not stable for its URL")
    source = retrieval_store.SourceMetadata(
        source_id=rule.source_id,
        source_name=rule.source_name,
        source_url=str(staged.get("source_url") or canonical_url),
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
        title=str(entry.get("title") or canonical_url),
        text=text,
        source=source,
        published_at=entry.get("published_at"),
        fetched_at=entry.get("fetched_at"),
    )


def refresh_staging(
    manifest: Mapping[str, Any],
    registry: TrustedSourceRegistry,
    *,
    contact: str,
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
    timeout_seconds_per_document: float = 20.0,
    apply: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch due documents into staging and create, but never approve, a candidate."""

    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    due = [
        entry for entry in manifest.get("documents", ())
        if isinstance(entry, Mapping)
        and (_parse_datetime(entry.get("refresh_after")) or current) <= current
        and entry.get("status") in APPROVED_DOCUMENT_STATUSES | {"pending_fetch"}
    ]
    if not apply:
        return {
            "status": "would_refresh",
            "document_ids": [entry.get("canonical_document_id") for entry in due],
        }
    candidate = json.loads(json.dumps(manifest))
    candidate.pop("manifest_hash", None)
    candidate["manifest_id"] = f"candidate_{uuid.uuid4().hex}"
    candidate["created_at"] = retrieval_store.utc_now()
    candidate["review"] = {"status": "pending", "reviewer": None, "reviewed_at": None}
    by_id = {
        str(entry.get("canonical_document_id")): entry
        for entry in candidate.get("documents", ()) if isinstance(entry, dict)
    }
    failures: list[dict[str, str]] = []
    for entry in due:
        document_id = str(entry.get("canonical_document_id"))
        try:
            fetched = source_fetch.fetch_source_document(
                str(entry.get("canonical_url")),
                registry,
                contact=contact,
                deadline=time.monotonic() + timeout_seconds_per_document,
            )
            decision = registry.evaluate_url(fetched.canonical_url, purpose="fetch")
            if not decision.allowed or decision.rule is None:
                raise CorpusOperationError("Refresh canonical URL was rejected")
            source_fetch.stage_source_document(
                fetched,
                decision.rule,
                staging_dir=Path(corpus_dir) / "staging",
            )
            updated = by_id[document_id]
            updated.update({
                "canonical_document_id": retrieval_store.stable_document_id(fetched.canonical_url),
                "canonical_url": fetched.canonical_url,
                "title": fetched.title,
                "published_at": fetched.published_at,
                "fetched_at": fetched.fetched_at,
                "content_hash": fetched.content_hash,
                "archive_hash": fetched.archive_hash,
                "extraction_version": fetched.extraction_version,
                "etag": fetched.etag,
                "last_modified": fetched.last_modified,
                "refresh_after": (current + timedelta(days=30)).isoformat().replace("+00:00", "Z"),
                "retention_until": (current + timedelta(days=decision.rule.retention_days)).isoformat().replace("+00:00", "Z"),
                "status": "pending_review",
                "expected_point_count": len(retrieval_store.chunk_document_text(fetched.text)),
                "rights_review": {
                    "status": "pending",
                    "reviewer": None,
                    "reviewed_at": None,
                    "third_party_marked": False,
                },
            })
        except Exception as exc:
            failures.append({"document_id": document_id, "code": str(getattr(exc, "code", "refresh_failed"))})
    candidate["manifest_hash"] = manifest_hash(candidate)
    candidates = Path(corpus_dir) / "candidates"
    candidate_path = candidates / f"manifest-{candidate['manifest_hash']}.json"
    _atomic_json_write(candidate_path, candidate)
    diff = diff_manifests(manifest, candidate)
    return {
        "status": "staged_candidate",
        "candidate_path": str(candidate_path),
        "candidate_hash": candidate["manifest_hash"],
        "diff": diff.to_json(),
        "failures": failures,
    }


def exact_statistics(
    store: retrieval_store.QdrantCorpusStore,
    *,
    manifest: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    points = store.list_all_points()
    by_source: dict[str, dict[str, set[str] | int]] = {}
    documents: dict[str, list[retrieval_store.StoredCorpusPoint]] = {}
    contents: dict[str, set[str]] = {}
    expired_documents: set[str] = set()
    due_documents: set[str] = set()
    for point in points:
        payload = point.payload
        source_id = str(payload.get("source_id") or "unknown")
        document_id = str(payload.get("document_id") or "")
        content_hash = str(payload.get("content_hash") or "")
        bucket = by_source.setdefault(source_id, {"documents": set(), "chunks": 0})
        assert isinstance(bucket["documents"], set)
        bucket["documents"].add(document_id)
        bucket["chunks"] = int(bucket["chunks"]) + 1
        documents.setdefault(document_id, []).append(point)
        contents.setdefault(content_hash, set()).add(document_id)
        retention = _parse_datetime(payload.get("retention_until"))
        if retention and retention <= current:
            expired_documents.add(document_id)
        elif retention and retention <= current + timedelta(days=30):
            due_documents.add(document_id)
    duplicates = {
        content_hash: sorted(ids) for content_hash, ids in contents.items()
        if content_hash and len(ids) > 1
    }
    mismatch: dict[str, Any] = {"missing_documents": [], "unexpected_documents": [], "point_count": {}}
    if manifest is not None:
        expected = {
            str(entry.get("canonical_document_id")): int(entry.get("expected_point_count") or 0)
            for entry in manifest.get("documents", ())
            if isinstance(entry, Mapping) and entry.get("status") in APPROVED_DOCUMENT_STATUSES
        }
        actual_ids = set(documents)
        mismatch["missing_documents"] = sorted(set(expected) - actual_ids)
        mismatch["unexpected_documents"] = sorted(actual_ids - set(expected))
        mismatch["point_count"] = {
            document_id: {"expected": expected[document_id], "actual": len(documents.get(document_id, ())) }
            for document_id in sorted(set(expected) & actual_ids)
            if expected[document_id] != len(documents.get(document_id, ()))
        }
    info = store.collection_info()
    capacity = _collection_capacity(info)
    return {
        "schema_version": "corpus_statistics/v1",
        "generated_at": retrieval_store.utc_now(),
        "collection": store.config.collection,
        "collection_schema": _collection_schema_summary(info),
        "capacity": capacity,
        "document_count": len(documents),
        "chunk_count": len(points),
        "by_source": {
            source: {"documents": len(value["documents"]), "chunks": value["chunks"]}
            for source, value in sorted(by_source.items())
        },
        "duplicates": duplicates,
        "expired_documents": sorted(expired_documents),
        "due_within_30_days": sorted(due_documents),
        "manifest_mismatches": mismatch,
    }


def expiry_plan(
    manifest: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> tuple[str, ...]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expired: list[str] = []
    for entry in manifest.get("documents", ()):
        if not isinstance(entry, Mapping):
            continue
        retention = _parse_datetime(entry.get("retention_until"))
        if retention and retention <= current:
            expired.append(str(entry.get("canonical_document_id") or ""))
    return tuple(sorted(filter(None, expired)))


def apply_expiry(
    store: retrieval_store.QdrantCorpusStore,
    manifest: Mapping[str, Any],
    *,
    backup_dir: Path | str,
    apply: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    if (manifest.get("review") or {}).get("status") != "approved":
        raise CorpusOperationError("Expiry deletion requires a reviewed manifest")
    document_ids = expiry_plan(manifest, now=now)
    result = {"status": "would_delete" if not apply else "deleted", "document_ids": list(document_ids), "point_count": 0}
    if not apply or not document_ids:
        return result
    require_recent_verified_backup(backup_dir, collection=store.config.collection, now=now)
    result["point_count"] = store.delete_documents(document_ids)
    return result


def create_verified_backup(
    store: retrieval_store.QdrantCorpusStore,
    *,
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
    manifest_dir: Path | str = DEFAULT_MANIFEST_DIR,
    generations: int = DEFAULT_BACKUP_GENERATIONS,
    apply: bool = False,
) -> dict[str, Any]:
    if not apply:
        return {"status": "would_backup", "collection": store.config.collection}
    root = Path(corpus_dir) / "backups"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generation = root / timestamp
    generation.mkdir()
    snapshot_name = store.create_snapshot()
    snapshot_path = store.download_snapshot(snapshot_name, generation / snapshot_name)
    archive_path = generation / "corpus-archive.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        manifests = Path(manifest_dir)
        if manifests.exists():
            archive.add(manifests, arcname="corpus_manifests", recursive=True)
        staging = Path(corpus_dir) / "staging"
        if staging.exists():
            archive.add(staging, arcname="staging", recursive=True)
    stats = exact_statistics(store)
    record = BackupRecord(
        schema_version="corpus_backup/v1",
        backup_id=f"backup_{timestamp}_{uuid.uuid4().hex[:8]}",
        created_at=retrieval_store.utc_now(),
        collection=store.config.collection,
        snapshot_name=snapshot_name,
        snapshot_path=str(snapshot_path),
        snapshot_sha256=_sha256_file(snapshot_path),
        archive_path=str(archive_path),
        archive_sha256=_sha256_file(archive_path),
        verified=snapshot_path.stat().st_size > 0 and archive_path.stat().st_size > 0,
        point_count=int(stats["chunk_count"]),
        qdrant_server_version=store.server_version(),
    )
    _atomic_json_write(generation / "backup.json", record.to_json())
    _prune_backup_generations(root, generations=generations)
    snapshots = sorted(
        store.list_snapshots(),
        key=lambda value: str(_value(value, "creation_time", "") or _value(value, "name", "")),
        reverse=True,
    )
    for obsolete in snapshots[generations:]:
        name = str(_value(obsolete, "name", "") or "")
        if name:
            store.delete_snapshot(name)
    return {"status": "backed_up", **record.to_json()}


def require_recent_verified_backup(
    backup_dir: Path | str,
    *,
    collection: str,
    now: datetime | None = None,
    maximum_age_days: int = 8,
) -> BackupRecord:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    records: list[BackupRecord] = []
    for path in Path(backup_dir).glob("*/backup.json"):
        try:
            value = load_json_object(path)
            record = BackupRecord(**value)
        except (CorpusOperationError, TypeError):
            continue
        created = _parse_datetime(record.created_at)
        if record.collection == collection and record.verified and created and current - created <= timedelta(days=maximum_age_days):
            if (
                Path(record.snapshot_path).is_file()
                and Path(record.archive_path).is_file()
                and _sha256_file(Path(record.snapshot_path)) == record.snapshot_sha256
                and _sha256_file(Path(record.archive_path)) == record.archive_sha256
            ):
                records.append(record)
    if not records:
        raise CorpusOperationError("A recent verified backup is required before deletion")
    return max(records, key=lambda item: item.created_at)


def verify_restore(
    store: retrieval_store.QdrantCorpusStore,
    backup: BackupRecord,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    collection = f"{store.config.collection}_restore_{uuid.uuid4().hex[:10]}"
    if not apply:
        return {"status": "would_restore_verify", "temporary_collection": collection}
    if _sha256_file(Path(backup.snapshot_path)) != backup.snapshot_sha256:
        raise CorpusOperationError("Snapshot hash verification failed before restore")
    current_version = store.server_version()
    if (
        backup.qdrant_server_version
        and current_version
        and _major_minor(backup.qdrant_server_version) != _major_minor(current_version)
    ):
        raise CorpusOperationError(
            "Snapshot restore verification requires a matching Qdrant major/minor version"
        )
    store.restore_snapshot_to_collection(backup.snapshot_path, collection)
    try:
        isolated_config = replace(store.config, collection=collection)
        isolated = retrieval_store.QdrantCorpusStore(
            isolated_config,
            client=store._get_client(),  # shared authenticated Cloud client
            models_module=store._models,
        )
        actual = len(isolated.list_all_points())
        if actual != backup.point_count:
            raise CorpusOperationError(
                f"Restored collection has {actual} points; expected {backup.point_count}"
            )
        return {"status": "verified", "point_count": actual, "temporary_collection": collection}
    finally:
        store.delete_collection(collection)


def write_run_record(
    operation: str,
    result: Mapping[str, Any],
    *,
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
) -> Path:
    root = Path(corpus_dir) / "runs"
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": f"run_{uuid.uuid4().hex}",
        "operation": operation,
        "created_at": retrieval_store.utc_now(),
        "result": dict(result),
    }
    path = root / f"{record['run_id']}.json"
    _atomic_json_write(path, record)
    return path


def install_scheduler_tasks(
    *,
    python_executable: str,
    script_path: Path | str,
    env_file: Path | str,
    apply: bool = False,
) -> dict[str, Any]:
    tasks = {
        "BiasCorpus-Health": ("DAILY", "5", "health-statistics"),
        "BiasCorpus-Weekly": ("WEEKLY", "1", "weekly-maintenance"),
        "BiasCorpus-RestoreVerify": ("MONTHLY", "1", "monthly-restore-verify"),
    }
    if not apply:
        return {"status": "would_install", "tasks": sorted(tasks)}
    for name, (schedule, modifier, command) in tasks.items():
        action = f'"{python_executable}" "{Path(script_path)}" {command} --env-file "{Path(env_file)}" --apply'
        subprocess.run(
            [
                "schtasks.exe", "/Create", "/F", "/TN", name, "/SC", schedule,
                "/MO", modifier, "/TR", action,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    return {"status": "installed", "tasks": sorted(tasks)}


def remove_scheduler_tasks(*, apply: bool = False) -> dict[str, Any]:
    names = ("BiasCorpus-Health", "BiasCorpus-Weekly", "BiasCorpus-RestoreVerify")
    if not apply:
        return {"status": "would_remove", "tasks": list(names)}
    for name in names:
        subprocess.run(
            ["schtasks.exe", "/Delete", "/F", "/TN", name],
            check=False,
            capture_output=True,
            text=True,
        )
    return {"status": "removed", "tasks": list(names)}


def scheduler_status() -> dict[str, Any]:
    names = ("BiasCorpus-Health", "BiasCorpus-Weekly", "BiasCorpus-RestoreVerify")
    status: dict[str, bool] = {}
    for name in names:
        result = subprocess.run(
            ["schtasks.exe", "/Query", "/TN", name],
            check=False,
            capture_output=True,
            text=True,
        )
        status[name] = result.returncode == 0
    return {"status": "inspected", "tasks": status}


def qdrant_cloud_store() -> retrieval_store.QdrantCorpusStore:
    config = retrieval_store.QdrantCorpusConfig.from_environment()
    if config.mode != "remote" or not config.api_key:
        raise CorpusOperationError("QDRANT_URL and QDRANT_API_KEY are required; local Qdrant is unsupported")
    if urlsplit(str(config.url)).scheme.casefold() != "https":
        raise CorpusOperationError("Lifecycle operations require an HTTPS Qdrant Cloud URL")
    return retrieval_store.QdrantCorpusStore(config)


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.casefold())


def _parse_datetime(value: Any) -> datetime | None:
    try:
        normalized = retrieval_store.normalise_datetime(str(value or ""))
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")) if normalized else None
    except (TypeError, ValueError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _major_minor(version: str) -> tuple[int, int] | None:
    try:
        parts = str(version).split(".")
        return int(parts[0]), int(parts[1])
    except (IndexError, TypeError, ValueError):
        return None


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prune_backup_generations(root: Path, *, generations: int) -> None:
    if generations <= 0:
        raise ValueError("Backup generations must be positive")
    directories = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)
    for obsolete in directories[generations:]:
        resolved = obsolete.resolve()
        if root.resolve() not in resolved.parents:
            raise CorpusOperationError("Refusing to prune outside the backup root")
        shutil.rmtree(resolved)


def _collection_schema_summary(info: Any) -> dict[str, Any] | None:
    if info is None:
        return None
    schema = retrieval_store.CorpusCollectionSchema(
        vector_size=retrieval_store.DEFAULT_EMBEDDING_DIMENSION,
        vector_name=retrieval_store.DEFAULT_VECTOR_NAME,
        content_field=retrieval_store.DEFAULT_CONTENT_FIELD,
        embedding_model=retrieval_store.DEFAULT_EMBEDDING_MODEL,
    )
    payload_schema = _value(info, "payload_schema", {}) or {}
    indexed_fields = set(payload_schema) if isinstance(payload_schema, Mapping) else set()
    expected_indexes = {value.field_name for value in schema.payload_indexes}
    strict_mode = _nested_value(info, "config", "strict_mode_config")
    return {
        "present": True,
        "compatible": schema.validation_error(info) is None,
        "validation_error": schema.validation_error(info),
        "indexed_fields": sorted(indexed_fields),
        "missing_indexes": sorted(expected_indexes - indexed_fields) if indexed_fields else [],
        "strict_mode": str(strict_mode)[:1000] if strict_mode is not None else None,
    }


def _collection_capacity(info: Any) -> dict[str, Any]:
    names = ("points_count", "indexed_vectors_count", "vectors_count", "segments_count")
    values = {
        name: (info.get(name) if isinstance(info, Mapping) else getattr(info, name, None))
        for name in names
    } if info is not None else {}
    if not values:
        return values
    points = values.get("points_count")
    maximum = _nested_value(info, "config", "strict_mode_config", "max_points_count")
    values["configured_max_points"] = maximum
    if isinstance(points, int):
        values["estimated_vector_bytes"] = (
            points * retrieval_store.DEFAULT_EMBEDDING_DIMENSION * 4
        )
    if isinstance(points, int) and isinstance(maximum, int) and maximum > 0:
        values["point_capacity_fraction"] = points / maximum
    return values


def _value(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _nested_value(value: Any, *names: str) -> Any:
    current = value
    for name in names:
        current = _value(current, name)
        if current is None:
            return None
    return current


LIFECYCLE_COMMANDS = {
    "readiness", "manifest-validate", "manifest-approve", "batch", "refresh",
    "expiry", "stats", "health", "backup", "restore-verify", "scheduler",
    "weekly-maintenance", "monthly-restore-verify", "health-statistics", "live-test",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trusted corpus and Qdrant Cloud lifecycle operations")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--env-file", type=Path)
    common.add_argument("--apply", action="store_true", help="Apply the reviewed mutation")
    common.add_argument(
        "--trusted-sources-file",
        type=Path,
        default=Path(__file__).with_name("trusted_sources.json"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("readiness", parents=[common])

    validate = commands.add_parser("manifest-validate", parents=[common])
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--require-approved", action="store_true")

    approve = commands.add_parser("manifest-approve", parents=[common])
    approve.add_argument("manifest", type=Path)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)

    batch = commands.add_parser("batch", parents=[common])
    batch.add_argument("--manifest", type=Path)
    batch.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    batch.add_argument("--staging-dir", type=Path, default=DEFAULT_CORPUS_DIR / "staging")

    refresh = commands.add_parser("refresh", parents=[common])
    refresh.add_argument("--manifest", type=Path)
    refresh.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    refresh.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)

    expiry = commands.add_parser("expiry", parents=[common])
    expiry.add_argument("--manifest", type=Path)
    expiry.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    expiry.add_argument("--backup-dir", type=Path, default=DEFAULT_CORPUS_DIR / "backups")

    stats = commands.add_parser("stats", parents=[common])
    stats.add_argument("--manifest", type=Path)
    stats.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)

    commands.add_parser("health", parents=[common])
    commands.add_parser("health-statistics", parents=[common])

    backup = commands.add_parser("backup", parents=[common])
    backup.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    backup.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    backup.add_argument("--generations", type=int, default=DEFAULT_BACKUP_GENERATIONS)

    restore = commands.add_parser("restore-verify", parents=[common])
    restore.add_argument("--backup-dir", type=Path, default=DEFAULT_CORPUS_DIR / "backups")

    scheduler = commands.add_parser("scheduler", parents=[common])
    scheduler.add_argument("action", choices=("install", "remove", "status"))

    weekly = commands.add_parser("weekly-maintenance", parents=[common])
    weekly.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    weekly.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)

    monthly = commands.add_parser("monthly-restore-verify", parents=[common])
    monthly.add_argument("--backup-dir", type=Path, default=DEFAULT_CORPUS_DIR / "backups")

    live = commands.add_parser("live-test", parents=[common])
    live.add_argument("provider", choices=("brave", "qdrant", "hybrid"))
    return parser.parse_args(argv)


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    registry = TrustedSourceRegistry.from_file(args.trusted_sources_file)
    command = args.command
    if command == "readiness":
        return {"status": "ready" if runtime_config.check_runtime_readiness().ready else "not_ready", **runtime_config.check_runtime_readiness().to_json()}
    if command == "manifest-validate":
        validation = validate_manifest(
            load_json_object(args.manifest),
            registry=registry,
            require_approved=args.require_approved,
        )
        return {"status": "valid" if validation.valid else "invalid", **validation.to_json()}
    if command == "manifest-approve":
        return approve_manifest(
            args.manifest,
            registry=registry,
            reviewer=args.reviewer,
            manifest_dir=args.manifest_dir,
            apply=args.apply,
        )
    if command == "scheduler":
        if args.action == "status":
            return scheduler_status()
        if args.action == "remove":
            return remove_scheduler_tasks(apply=args.apply)
        if not args.env_file:
            raise CorpusOperationError("Scheduler installation requires an explicit --env-file")
        return install_scheduler_tasks(
            python_executable=sys.executable,
            script_path=Path(__file__).with_name("ingest_trusted_corpus.py"),
            env_file=args.env_file,
            apply=args.apply,
        )
    if command == "live-test":
        from live_provider_checks import run_live_checks

        return run_live_checks(command=args.provider, apply=args.apply, policy_file=args.trusted_sources_file)

    store = qdrant_cloud_store()
    if command in {"health", "health-statistics"}:
        health = store.health()
        result = {
            "status": health.status,
            "reason": health.reason,
            "collection": store.config.collection,
            "collection_present": store.collection_info() is not None if health.status == "ready" else False,
        }
        if command == "health-statistics" and health.status == "ready":
            result["statistics"] = exact_statistics(store)
        return result
    if command in {"batch", "refresh", "expiry", "stats"}:
        manifest = load_json_object(args.manifest) if args.manifest else load_current_manifest(args.manifest_dir)
    if command == "batch":
        return batch_ingest_manifest(
            store,
            manifest,
            registry,
            staging_dir=args.staging_dir,
            apply=args.apply,
        )
    if command == "refresh":
        contact = str(os.environ.get("CORPUS_FETCH_CONTACT") or "")
        if args.apply and not contact:
            raise CorpusOperationError("CORPUS_FETCH_CONTACT is required for refresh")
        return refresh_staging(
            manifest,
            registry,
            contact=contact,
            corpus_dir=args.corpus_dir,
            apply=args.apply,
        )
    if command == "expiry":
        return apply_expiry(
            store,
            manifest,
            backup_dir=args.backup_dir,
            apply=args.apply,
        )
    if command == "stats":
        return exact_statistics(store, manifest=manifest)
    if command == "backup":
        return create_verified_backup(
            store,
            corpus_dir=args.corpus_dir,
            manifest_dir=args.manifest_dir,
            generations=args.generations,
            apply=args.apply,
        )
    if command in {"restore-verify", "monthly-restore-verify"}:
        backup = _latest_backup_record(args.backup_dir, collection=store.config.collection)
        return verify_restore(store, backup, apply=args.apply)
    if command == "weekly-maintenance":
        manifest = load_current_manifest(args.manifest_dir)
        contact = str(os.environ.get("CORPUS_FETCH_CONTACT") or "")
        if args.apply and not contact:
            raise CorpusOperationError("CORPUS_FETCH_CONTACT is required for weekly refresh")
        return {
            "status": "maintained" if args.apply else "would_maintain",
            "refresh": refresh_staging(
                manifest, registry, contact=contact, corpus_dir=args.corpus_dir, apply=args.apply
            ),
            "backup": create_verified_backup(
                store,
                corpus_dir=args.corpus_dir,
                manifest_dir=args.manifest_dir,
                apply=args.apply,
            ),
            "expiry_dry_run": apply_expiry(
                store,
                manifest,
                backup_dir=Path(args.corpus_dir) / "backups",
                apply=False,
            ),
        }
    raise CorpusOperationError(f"Unsupported lifecycle command: {command}")


def _latest_backup_record(backup_dir: Path | str, *, collection: str) -> BackupRecord:
    records: list[BackupRecord] = []
    for path in Path(backup_dir).glob("*/backup.json"):
        try:
            record = BackupRecord(**load_json_object(path))
        except (CorpusOperationError, TypeError):
            continue
        if record.collection == collection and record.verified:
            records.append(record)
    if not records:
        raise CorpusOperationError("No verified backup is available")
    return max(records, key=lambda value: value.created_at)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        runtime_config.load_explicit_env_file(args.env_file)
        result = run_command(args)
        if args.apply and args.command not in {"readiness", "manifest-validate", "stats", "health"}:
            try:
                result["run_record"] = str(write_run_record(args.command, result))
            except OSError:
                pass
    except (
        CorpusOperationError,
        runtime_config.RuntimeConfigurationError,
        retrieval_store.CorpusStoreError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") not in {"error", "invalid", "not_ready", "unavailable"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
