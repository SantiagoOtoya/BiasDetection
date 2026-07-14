#!/usr/bin/env python
"""Explicitly gated live Brave and isolated Qdrant Cloud acceptance checks."""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import evidence_retrieval
import retrieval_store
import runtime_config


class LiveCheckError(RuntimeError):
    pass


LIVE_COLLECTION_PREFIX = "trusted_corpus_live_"


def run_live_checks(*, command: str, apply: bool, policy_file: Path | str) -> dict[str, Any]:
    if not apply:
        return {"status": "gated", "provider": command, "reason": "Pass --apply to run live checks"}
    runtime_config.require_runtime_readiness(
        "hybrid" if command == "hybrid" else "web" if command == "brave" else "corpus"
    )
    results: dict[str, Any] = {}
    if command in {"brave", "hybrid"}:
        results["brave"] = check_brave(policy_file)
    if command in {"qdrant", "hybrid"}:
        results["qdrant"] = check_qdrant()
    return {"status": "passed", "provider": command, "checks": results}


def check_brave(policy_file: Path | str) -> dict[str, Any]:
    key = str(os.environ.get("BRAVE_SEARCH_API_KEY") or "")
    if not key:
        raise LiveCheckError("BRAVE_SEARCH_API_KEY is required")
    policy = evidence_retrieval.load_trusted_source_policy(policy_file)
    trusted = evidence_retrieval.search_brave(
        evidence_retrieval.EvidenceQuery(
            text="site:bls.gov Consumer Price Index",
            source="live_check",
            claim_ids=("live-trusted",),
        ),
        key,
        10.0,
        policy,
    )
    if not trusted or not all(evidence_retrieval.is_permitted_web_url(item.url, policy) for item in trusted):
        raise LiveCheckError("Live Brave trusted site: query did not return safe results")
    diagnostics = getattr(trusted, "diagnostics", {})
    if not diagnostics.get("rate_limit"):
        raise LiveCheckError("Live Brave response omitted documented rate-limit headers")
    nonce = uuid.uuid4().hex
    empty = evidence_retrieval.search_brave(
        evidence_retrieval.EvidenceQuery(
            text=f"site:bls.gov codex-no-result-{nonce}",
            source="live_check",
            claim_ids=("live-empty",),
        ),
        key,
        10.0,
        policy,
    )
    nonce_empty = not empty
    warnings = [] if nonce_empty else [{
        "code": "nonce_nonempty_warning",
        "message": "Brave returned results for the nonce query; empty-result behavior is not deterministic",
        "result_count": len(empty),
    }]
    return {
        "authentication_status": "passed",
        "trusted_query_status": "passed",
        "trusted_result_count": len(trusted),
        "nonce_result_count": len(empty),
        "nonce_status": "none_found" if nonce_empty else "nonempty_warning",
        "rate_headers_present": bool(diagnostics.get("rate_limit")),
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def check_qdrant() -> dict[str, Any]:
    base = retrieval_store.QdrantCorpusConfig.from_environment()
    if base.mode != "remote" or not base.api_key:
        raise LiveCheckError("QDRANT_URL and QDRANT_API_KEY are required")
    base_store = retrieval_store.QdrantCorpusStore(base)
    before = set(base_store.list_collection_names())
    existing_temporary = sorted(
        name for name in before if name.startswith(LIVE_COLLECTION_PREFIX)
    )
    if existing_temporary:
        raise LiveCheckError(
            "A pre-existing live-test Qdrant collection requires manual review; none was modified"
        )
    production_present_before = retrieval_store.DEFAULT_COLLECTION in before
    collection = f"{LIVE_COLLECTION_PREFIX}{uuid.uuid4().hex[:12]}"
    config = replace(base, collection=collection)
    store = retrieval_store.QdrantCorpusStore(config)
    source = retrieval_store.SourceMetadata(
        source_id="us_bls",
        source_name="U.S. Bureau of Labor Statistics",
        source_url="https://www.bls.gov/live-provider-check",
        canonical_url="https://www.bls.gov/live-provider-check",
        source_domain="bls.gov",
        source_tier="primary_official",
        source_policy_version="trusted_sources/v2",
        license_id="us_bls_public_domain_text",
        license_url="https://www.bls.gov/opub/copyright-information.htm",
        redistribution_allowed=True,
        retention_policy="review_after_365_days",
        retention_days=365,
    )
    vector = [1.0] + [0.0] * 383
    result: dict[str, Any] | None = None
    operation_error: Exception | None = None
    try:
        schema = retrieval_store.CorpusCollectionSchema(
            vector_size=384,
            vector_name=retrieval_store.DEFAULT_VECTOR_NAME,
            content_field=retrieval_store.DEFAULT_CONTENT_FIELD,
            embedding_model=retrieval_store.DEFAULT_EMBEDDING_MODEL,
        )
        store.ensure_collection(schema)
        first_document = retrieval_store.CorpusDocument(
            title="Live provider check",
            text="A deterministic live provider check document for isolated Qdrant operations.",
            source=source,
            fetched_at=retrieval_store.utc_now(),
        )
        first = store.ingest_document(first_document, [vector])
        hits = store.search_points(vector, limit=3)
        filtered = store.list_all_points(
            retrieval_store.CorpusMetadataFilter(source_ids=("us_bls",))
        )
        replacement = retrieval_store.CorpusDocument(
            title="Live provider replacement",
            text="A changed deterministic document verifies reviewed replacement behavior.",
            source=source,
            fetched_at=retrieval_store.utc_now(),
        )
        second = store.ingest_document(replacement, [vector])
        document_id = retrieval_store.stable_document_id(source.canonical_url)
        deleted = store.delete_documents((document_id,))
        snapshot_name = store.create_snapshot()
        with tempfile.TemporaryDirectory() as directory:
            snapshot_path = store.download_snapshot(
                snapshot_name,
                Path(directory) / snapshot_name,
            )
            snapshot_size = snapshot_path.stat().st_size
        if first.status != "ingested" or second.status != "replaced" or not hits or not filtered or deleted <= 0:
            raise LiveCheckError("One or more isolated Qdrant operations failed")
        result = {
            "collection": collection,
            "ingest": first.status,
            "query_count": len(hits),
            "filter_count": len(filtered),
            "replacement": second.status,
            "deleted_points": deleted,
            "snapshot_created": bool(snapshot_name),
            "snapshot_downloaded_bytes": snapshot_size,
        }
    except Exception as exc:
        operation_error = exc

    cleanup_error: Exception | None = None
    try:
        store.delete_collection(collection)
    except Exception as exc:
        cleanup_error = exc

    after: set[str] = set()
    try:
        after = set(base_store.list_collection_names())
    except Exception as exc:
        cleanup_error = cleanup_error or exc

    remaining_temporary = sorted(
        name for name in after if name.startswith(LIVE_COLLECTION_PREFIX)
    )
    production_present_after = retrieval_store.DEFAULT_COLLECTION in after
    cleanup_confirmed = (
        cleanup_error is None
        and collection not in after
        and not remaining_temporary
        and production_present_after == production_present_before
    )
    if not cleanup_confirmed:
        raise LiveCheckError(
            "Qdrant live-test cleanup could not confirm an unchanged cluster state"
        ) from cleanup_error or operation_error
    if operation_error is not None:
        raise operation_error
    if result is None:
        raise LiveCheckError("Qdrant live test produced no result")
    result.update({
        "cleanup_confirmed": True,
        "remaining_temporary_collections": remaining_temporary,
        "production_collection_present_before": production_present_before,
        "production_collection_present_after": production_present_after,
    })
    return result
