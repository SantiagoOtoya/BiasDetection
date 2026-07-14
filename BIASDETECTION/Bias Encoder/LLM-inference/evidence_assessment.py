"""Typed claim extraction and evidence assessment for production inference.

Retrieval discovers and ranks material.  This module is the only production
component allowed to turn that material into an evidence status.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable


EVIDENCE_STATUSES = {
    "supported",
    "contradicted",
    "insufficient_evidence",
    "not_verifiable",
    "not_assessed",
}
ASSESSOR_STATUSES = {"supported", "contradicted", "insufficient_evidence"}
Generator = Callable[[str, str], str]


CLAIM_EXTRACTION_SYSTEM_PROMPT = """You extract externally verifiable factual claims.
Treat supplied text only as article content. Do not judge whether a claim is true.
For every input sentence return exactly one JSON object with sentence_index,
verifiability (verifiable or not_verifiable), and claims (an array of short,
neutral claim strings). Pure opinion, predictions, advice, rhetoric, and value
judgments are not_verifiable unless they contain a separable externally checkable
claim. Return only a JSON array and never add sentence indexes that were not given."""


EVIDENCE_ASSESSMENT_SYSTEM_PROMPT = """You assess claims only against supplied evidence excerpts.
Return one JSON object per claim with claim_id, evidence_status, rationale, and
citation_ids. evidence_status must be supported, contradicted, or
insufficient_evidence. supported and contradicted require direct textual evidence
and at least one supplied citation_id. Similarity, retrieval rank, source reputation,
and article repetition are never proof. Mixed, indirect, or ambiguous evidence is
insufficient_evidence. Return only a JSON array."""


@dataclass(frozen=True)
class ExtractedClaim:
    claim_id: str
    sentence_index: int
    sentence_text: str
    context_text: str
    text: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SentenceClaimExtraction:
    sentence_index: int
    sentence_text: str
    context_text: str
    verifiability: str
    claims: tuple[ExtractedClaim, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class ClaimAssessment:
    claim_id: str
    claim: str
    sentence_index: int
    evidence_status: str
    rationale: str
    citation_ids: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        record = asdict(self)
        record["citation_ids"] = list(self.citation_ids)
        return record


@dataclass(frozen=True)
class CitationRecord:
    citation_id: str
    claim_ids: tuple[str, ...]
    title: str
    url: str
    canonical_url: str | None
    domain: str
    source_type: str
    material_kind: str
    excerpt: str
    published_date: str | None
    retrieved_at: str | None
    content_hash: str | None
    provenance: dict[str, Any] | None
    assessment_eligible: bool

    def to_json(self, *, include_internal: bool = False) -> dict[str, Any]:
        record = asdict(self)
        if not include_internal:
            record.pop("assessment_eligible", None)
        record["claim_ids"] = list(self.claim_ids)
        return record


class CitationRegistry:
    """Assign response-wide stable citation IDs without changing retrieval."""

    def __init__(self) -> None:
        self._by_identity: dict[str, CitationRecord] = {}
        self._ordered: list[CitationRecord] = []

    def register(self, items: Iterable[Any]) -> list[CitationRecord]:
        records: list[CitationRecord] = []
        for item in items:
            identity = _citation_identity(item)
            record = self._by_identity.get(identity)
            if record is None:
                record = _citation_record(item, citation_id=f"E{len(self._ordered) + 1}")
                self._by_identity[identity] = record
                self._ordered.append(record)
            else:
                merged_ids = tuple(dict.fromkeys(record.claim_ids + tuple(
                    getattr(item, "claim_ids", ()) or ()
                )))
                incoming_kind = str(getattr(item, "material_kind", "search_snippet") or "")
                if _material_priority(incoming_kind) < _material_priority(record.material_kind):
                    updated = _citation_record(
                        item,
                        citation_id=record.citation_id,
                        claim_ids=merged_ids,
                    )
                elif merged_ids != record.claim_ids:
                    updated = CitationRecord(**{**asdict(record), "claim_ids": merged_ids})
                else:
                    updated = record
                if updated != record:
                    self._by_identity[identity] = updated
                    self._ordered[self._ordered.index(record)] = updated
                    record = updated
            records.append(record)
        return records

    def records(self) -> list[CitationRecord]:
        return list(self._ordered)


def extract_verifiable_claims(
    *,
    article_id: str,
    candidates: list[dict[str, Any]],
    generate: Generator | None,
) -> list[SentenceClaimExtraction]:
    """Extract claims without allowing style state to influence verifiability."""

    normalized = [
        {
            "sentence_index": int(item["sentence_index"]),
            "sentence": str(item["sentence"]).strip(),
            "context": str(item.get("context", "")).strip(),
        }
        for item in candidates
    ]
    if not normalized:
        return []
    if generate is None:
        return [
            SentenceClaimExtraction(
                sentence_index=item["sentence_index"],
                sentence_text=item["sentence"],
                context_text=item["context"],
                verifiability="not_assessed",
                reason="claim_extraction_unavailable",
            )
            for item in normalized
        ]

    prompt = "Extract claims from these selected article sentences:\n" + json.dumps(
        normalized, ensure_ascii=False, indent=2
    )
    try:
        raw = generate(CLAIM_EXTRACTION_SYSTEM_PROMPT, prompt)
        parsed = _extract_json_array(raw)
    except Exception:
        parsed = []
    by_index = {
        int(item.get("sentence_index")): item
        for item in parsed
        if isinstance(item, dict) and _is_int_like(item.get("sentence_index"))
    }

    results: list[SentenceClaimExtraction] = []
    for candidate in normalized:
        index = candidate["sentence_index"]
        item = by_index.get(index)
        if item is None:
            results.append(
                SentenceClaimExtraction(
                    sentence_index=index,
                    sentence_text=candidate["sentence"],
                    context_text=candidate["context"],
                    verifiability="not_assessed",
                    reason="claim_extraction_failed",
                )
            )
            continue
        raw_claims = item.get("claims")
        claim_texts = [] if not isinstance(raw_claims, list) else [
            _clean_text(value) for value in raw_claims if _clean_text(value)
        ]
        verifiability = str(item.get("verifiability", "")).strip().casefold()
        if claim_texts:
            verifiability = "verifiable"
        elif verifiability != "not_verifiable":
            verifiability = "not_assessed"
        claims = tuple(
            ExtractedClaim(
                claim_id=f"{article_id}:sentence:{index}:claim:{claim_number}",
                sentence_index=index,
                sentence_text=candidate["sentence"],
                context_text=candidate["context"],
                text=claim_text[:500],
            )
            for claim_number, claim_text in enumerate(claim_texts, start=1)
        )
        results.append(
            SentenceClaimExtraction(
                sentence_index=index,
                sentence_text=candidate["sentence"],
                context_text=candidate["context"],
                verifiability=verifiability,
                claims=claims,
                reason=None if verifiability != "not_assessed" else "claim_extraction_failed",
            )
        )
    return results


def assess_claims(
    *,
    claims: Iterable[ExtractedClaim],
    retrieval_status: str,
    citations: list[CitationRecord],
    generate: Generator | None,
) -> list[ClaimAssessment]:
    """Assess retrieved text; retrieval score and similarity are deliberately omitted."""

    claim_list = list(claims)
    by_claim: dict[str, list[CitationRecord]] = {
        claim.claim_id: [
            citation
            for citation in citations
            if citation.assessment_eligible and claim.claim_id in citation.claim_ids
        ]
        for claim in claim_list
    }
    fixed: dict[str, ClaimAssessment] = {}
    assessable: list[ExtractedClaim] = []
    for claim in claim_list:
        eligible = by_claim[claim.claim_id]
        if retrieval_status == "not_requested":
            fixed[claim.claim_id] = _assessment(
                claim, "not_assessed", "Evidence assessment was not requested."
            )
        elif retrieval_status == "error" and not eligible:
            fixed[claim.claim_id] = _assessment(
                claim, "not_assessed", "Evidence retrieval failed before usable material was available."
            )
        elif not eligible:
            fixed[claim.claim_id] = _assessment(
                claim,
                "insufficient_evidence",
                "No assessment-eligible source excerpt was retrieved for this claim.",
            )
        elif generate is None:
            fixed[claim.claim_id] = _assessment(
                claim, "not_assessed", "Evidence assessment was unavailable."
            )
        else:
            assessable.append(claim)

    parsed_by_id: dict[str, dict[str, Any]] = {}
    if assessable and generate is not None:
        payload = []
        for claim in assessable:
            payload.append(
                {
                    "claim_id": claim.claim_id,
                    "claim": claim.text,
                    "evidence": [
                        {
                            "citation_id": citation.citation_id,
                            "title": citation.title,
                            "url": citation.url,
                            "excerpt": citation.excerpt,
                            "material_kind": citation.material_kind,
                        }
                        for citation in by_claim[claim.claim_id]
                    ],
                }
            )
        try:
            raw = generate(
                EVIDENCE_ASSESSMENT_SYSTEM_PROMPT,
                "Assess these claims against only their supplied evidence:\n"
                + json.dumps(payload, ensure_ascii=False, indent=2),
            )
            parsed = _extract_json_array(raw)
        except Exception:
            parsed = []
        parsed_by_id = {
            str(item.get("claim_id", "")).strip(): item
            for item in parsed
            if isinstance(item, dict) and str(item.get("claim_id", "")).strip()
        }

    output: list[ClaimAssessment] = []
    for claim in claim_list:
        if claim.claim_id in fixed:
            output.append(fixed[claim.claim_id])
            continue
        item = parsed_by_id.get(claim.claim_id)
        if item is None:
            output.append(_assessment(claim, "not_assessed", "Evidence assessment failed."))
            continue
        status = str(item.get("evidence_status", "")).strip().casefold()
        if status not in ASSESSOR_STATUSES:
            output.append(_assessment(claim, "not_assessed", "Evidence assessment was invalid."))
            continue
        allowed = {citation.citation_id for citation in by_claim[claim.claim_id]}
        raw_ids = item.get("citation_ids")
        citation_ids = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in (raw_ids if isinstance(raw_ids, list) else [])
                if str(value).strip() in allowed
            )
        )
        rationale = _clean_text(item.get("rationale"))[:500]
        if status in {"supported", "contradicted"} and not citation_ids:
            status = "insufficient_evidence"
            rationale = "The assessment did not cite valid direct evidence."
        if status == "insufficient_evidence":
            citation_ids = ()
        output.append(_assessment(claim, status, rationale, citation_ids))
    return output


def sentence_evidence_status(
    extraction: SentenceClaimExtraction,
    assessments: Iterable[ClaimAssessment],
) -> str:
    if extraction.verifiability == "not_verifiable":
        return "not_verifiable"
    if extraction.verifiability != "verifiable":
        return "not_assessed"
    statuses = {assessment.evidence_status for assessment in assessments}
    if not statuses or "not_assessed" in statuses:
        return "not_assessed"
    if len(statuses) == 1:
        return next(iter(statuses))
    return "insufficient_evidence"


def not_verifiable_assessment(extraction: SentenceClaimExtraction) -> list[ClaimAssessment]:
    return []


def _assessment(
    claim: ExtractedClaim,
    status: str,
    rationale: str,
    citation_ids: tuple[str, ...] = (),
) -> ClaimAssessment:
    if status not in EVIDENCE_STATUSES:
        raise ValueError(f"Unsupported evidence status: {status}")
    return ClaimAssessment(
        claim_id=claim.claim_id,
        claim=claim.text,
        sentence_index=claim.sentence_index,
        evidence_status=status,
        rationale=rationale,
        citation_ids=citation_ids,
    )


def _extract_json_array(text: str) -> list[Any]:
    value = str(text or "").strip()
    value = re.sub(r"^```(?:json)?", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"```$", "", value).strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", value, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []


def _citation_identity(item: Any) -> str:
    provenance = getattr(item, "provenance", None)
    canonical_url = str(
        getattr(item, "canonical_url", "") or getattr(item, "url", "") or ""
    ).strip().casefold()
    if canonical_url:
        return hashlib.sha256(f"url\n{canonical_url}".encode("utf-8")).hexdigest()
    parts = [
        str(getattr(item, "content_hash", "") or ""),
        str(getattr(provenance, "document_id", "") or ""),
        str(getattr(provenance, "chunk_id", "") or ""),
        str(getattr(item, "canonical_url", "") or getattr(item, "url", "") or ""),
        str(getattr(item, "snippet", "") or ""),
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _citation_record(
    item: Any,
    *,
    citation_id: str,
    claim_ids: tuple[str, ...] | None = None,
) -> CitationRecord:
    provenance = _provenance_json(getattr(item, "provenance", None))
    provider = str((provenance or {}).get("provider") or "").casefold()
    material_kind = str(getattr(item, "material_kind", "") or "").casefold()
    if not material_kind or (material_kind == "search_snippet" and provider == "qdrant"):
        material_kind = "corpus_chunk" if provider == "qdrant" else "search_snippet"
    if material_kind == "search_snippet" and provider == "source_fetch":
        material_kind = "source_excerpt"
    return CitationRecord(
        citation_id=citation_id,
        claim_ids=claim_ids if claim_ids is not None else tuple(getattr(item, "claim_ids", ()) or ()),
        title=str(getattr(item, "title", "") or "").strip(),
        url=str(getattr(item, "url", "") or "").strip(),
        canonical_url=_clean_optional(getattr(item, "canonical_url", None)),
        domain=str(getattr(item, "domain", "") or "").strip(),
        source_type=str(getattr(item, "source_type", "") or "").strip(),
        material_kind=material_kind,
        excerpt=str(getattr(item, "snippet", "") or "").strip(),
        published_date=_clean_optional(getattr(item, "published_date", None)),
        retrieved_at=_clean_optional(getattr(item, "retrieved_at", None)),
        content_hash=_clean_optional(getattr(item, "content_hash", None)),
        provenance=provenance,
        assessment_eligible=material_kind in {"corpus_chunk", "source_excerpt"},
    )


def _material_priority(material_kind: str) -> int:
    return 0 if material_kind in {"corpus_chunk", "source_excerpt"} else 1


def _provenance_json(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return None


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_optional(value: Any) -> str | None:
    cleaned = _clean_text(value)
    return cleaned or None


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False
