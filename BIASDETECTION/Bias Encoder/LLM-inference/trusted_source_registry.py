#!/usr/bin/env python
"""Versioned source, fetch, ingestion, retention, and rights policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import retrieval_store


POLICY_VERSION = "trusted_sources/v2"
REGISTRY_SCHEMA_VERSION = "trusted_corpus_sources/v2"


class SourceRegistryError(ValueError):
    """The trusted-source registry is invalid or rejects an operation."""


@dataclass(frozen=True)
class SourceRule:
    source_id: str
    source_name: str
    domains: tuple[str, ...]
    allowed_url_prefixes: tuple[str, ...]
    source_tier: str
    fetch_allowed: bool
    ingest_allowed: bool
    license_id: str
    license_url: str
    redistribution_allowed: bool
    retention_policy: str
    retention_days: int
    rights_review_required: bool = True
    excluded_url_prefixes: tuple[str, ...] = ()
    excluded_material: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "SourceRule":
        rights = data.get("rights") or data.get("license") or {}
        retention = data.get("retention") or {}
        review = data.get("per_document_rights_review") or {}
        rule = cls(
            source_id=_required(data, "source_id"),
            source_name=_required(data, "source_name"),
            domains=tuple(_domain(value) for value in data.get("domains", ()) if _domain(value)),
            allowed_url_prefixes=tuple(
                value
                for value in (_prefix(item) for item in data.get("allowed_url_prefixes", ()))
                if value
            ),
            source_tier=_required(data, "source_tier"),
            fetch_allowed=bool(data.get("fetch_allowed", data.get("ingest_allowed", False))),
            ingest_allowed=bool(data.get("ingest_allowed")),
            license_id=str(rights.get("id") or "").strip(),
            license_url=str(rights.get("basis_url") or rights.get("url") or "").strip(),
            redistribution_allowed=bool(rights.get("redistribution_allowed")),
            retention_policy=str(retention.get("policy") or "").strip(),
            retention_days=int(retention.get("days") or 0),
            rights_review_required=bool(review.get("required", True)),
            excluded_url_prefixes=tuple(
                value
                for value in (_prefix(item) for item in data.get("excluded_url_prefixes", ()))
                if value
            ),
            excluded_material=tuple(
                str(value).strip() for value in data.get("excluded_material", ()) if str(value).strip()
            ),
        )
        if not rule.domains or not rule.allowed_url_prefixes:
            raise SourceRegistryError(
                f"Source rule {rule.source_id!r} requires domains and allowed URL prefixes"
            )
        for name in ("license_id", "license_url", "retention_policy"):
            if not getattr(rule, name):
                raise SourceRegistryError(
                    f"Source rule {rule.source_id!r} requires {name}"
                )
        if rule.retention_days <= 0:
            raise SourceRegistryError(
                f"Source rule {rule.source_id!r} requires a positive retention period"
            )
        if rule.ingest_allowed and not rule.redistribution_allowed:
            raise SourceRegistryError(
                f"Source rule {rule.source_id!r} cannot allow ingestion without redistribution rights"
            )
        return rule


@dataclass(frozen=True)
class SourcePolicyDecision:
    allowed: bool
    canonical_url: str | None
    rule: SourceRule | None
    purpose: str
    reason: str | None = None


class TrustedSourceRegistry:
    """Strict allow-list with independent discovery, fetch, and ingest decisions."""

    def __init__(self, *, policy_version: str, rules: Sequence[SourceRule]) -> None:
        self.policy_version = str(policy_version or "").strip()
        self.rules = tuple(rules)
        if self.policy_version not in {"trusted_sources/v1", POLICY_VERSION}:
            raise SourceRegistryError(f"Unsupported source policy: {self.policy_version}")
        if not self.rules:
            raise SourceRegistryError("Trusted-source registry has no source rules")

    @classmethod
    def from_file(cls, path: Path | str) -> "TrustedSourceRegistry":
        source_path = Path(path)
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceRegistryError("Could not read a valid trusted-source registry") from exc
        if not isinstance(payload, Mapping):
            raise SourceRegistryError("Trusted-source registry must be a JSON object")
        registry = payload.get("corpus_registry")
        if not isinstance(registry, Mapping):
            raise SourceRegistryError("Trusted-source registry lacks corpus_registry")
        schema = str(registry.get("schema_version") or "")
        if schema not in {"trusted_corpus_sources/v1", REGISTRY_SCHEMA_VERSION}:
            raise SourceRegistryError(f"Unsupported corpus registry schema: {schema}")
        records = registry.get("sources")
        if not isinstance(records, list) or not all(isinstance(item, Mapping) for item in records):
            raise SourceRegistryError("corpus_registry.sources must be an array of objects")
        return cls(
            policy_version=str(payload.get("version") or "trusted_sources/v1"),
            rules=tuple(SourceRule.from_json(item) for item in records),
        )

    def evaluate_url(self, url: str, *, purpose: str = "ingest") -> SourcePolicyDecision:
        operation = str(purpose or "").strip().casefold()
        if operation not in {"discover", "fetch", "ingest"}:
            raise ValueError("purpose must be discover, fetch, or ingest")
        canonical = retrieval_store.canonicalize_document_url(url)
        if canonical is None:
            return SourcePolicyDecision(
                False, None, None, operation,
                "source URL must be absolute HTTPS without credentials",
            )
        domain = retrieval_store.normalise_domain(canonical)
        matches = [
            rule for rule in self.rules
            if any(domain == allowed or domain.endswith(f".{allowed}") for allowed in rule.domains)
            and any(canonical.casefold().startswith(prefix.casefold()) for prefix in rule.allowed_url_prefixes)
        ]
        if not matches:
            return SourcePolicyDecision(
                False, canonical, None, operation,
                "source is not explicitly permitted by the trusted-source registry",
            )
        rule = max(matches, key=lambda item: max(len(p) for p in item.allowed_url_prefixes))
        if any(canonical.casefold().startswith(prefix.casefold()) for prefix in rule.excluded_url_prefixes):
            return SourcePolicyDecision(
                False, canonical, rule, operation,
                "URL is explicitly excluded by the source policy",
            )
        allowed = (
            operation == "discover"
            or (operation == "fetch" and rule.fetch_allowed)
            or (operation == "ingest" and rule.ingest_allowed)
        )
        if not allowed:
            return SourcePolicyDecision(
                False, canonical, rule, operation,
                f"source '{rule.source_id}' is not permitted for {operation}",
            )
        return SourcePolicyDecision(True, canonical, rule, operation)

    def require_document_rights(
        self,
        url: str,
        *,
        reviewer: str | None,
        reviewed_at: str | None,
        third_party_marked: bool = False,
    ) -> SourceRule:
        decision = self.evaluate_url(url, purpose="ingest")
        if not decision.allowed or decision.rule is None:
            raise SourceRegistryError(decision.reason or "Document is not ingestible")
        rule = decision.rule
        if third_party_marked:
            raise SourceRegistryError("Third-party-marked material may not be ingested")
        if rule.rights_review_required and not (str(reviewer or "").strip() and str(reviewed_at or "").strip()):
            raise SourceRegistryError("Per-document rights reviewer and review date are required")
        return rule


def _required(data: Mapping[str, Any], name: str) -> str:
    value = str(data.get(name) or "").strip()
    if not value:
        raise SourceRegistryError(f"Source registry field {name!r} is required")
    return value


def _domain(value: Any) -> str:
    return retrieval_store.normalise_domain(str(value or ""))


def _prefix(value: Any) -> str | None:
    return retrieval_store.canonicalize_document_url(str(value or ""))
