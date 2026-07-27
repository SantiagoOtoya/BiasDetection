"""Pure mapping from a v3 ``analyze_article`` record to the backend API schema.

No v3-stack imports here: this module is deliberately stack-free so it can be
unit-tested in the same process as the v2/root test suite (importing the v3
stack would collide with root modules of the same name). ``v3_adapter``
delegates to these functions.
"""

from __future__ import annotations

from typing import Any

import fact_checker
import scoring
from schemas import AnalyzeResponse, SelectedSentence


def selected_sentence_from_v3(p: dict[str, Any]) -> SelectedSentence:
    return SelectedSentence(
        index=int(p.get("index", 0)),
        text=str(p.get("text", "")),
        bias_label=str(p.get("bias_assessment") or "clear_bias"),
        bias_probability=float(p.get("p_bias") or 0.0),
        opinion_label=str(p.get("opinion_style") or "uncertain"),
        opinion_probability=float(p.get("p_opinionated_style") or 0.0),
        selection_reasons=list(p.get("selection_reasons") or []),
        abstention_reasons=list(p.get("abstention_reasons") or []),
        bias_assessment=p.get("bias_assessment"),
        opinion_style=p.get("opinion_style"),
        uncertainty_status=p.get("uncertainty_status"),
    )


def combine_reports(record: dict[str, Any], selected: list[SelectedSentence]) -> str:
    if not selected:
        return (
            "No clearly biased wording was flagged in this page under the "
            "high-precision analysis. The passage reads as largely factual "
            "based on the text provided."
        )
    reports = [
        w.get("llm_report")
        for w in record.get("context_windows") or []
        if w.get("llm_report")
    ]
    if reports:
        return "\n\n".join(r.strip() for r in reports)
    lines = [
        "The full report model is not loaded in this mode, so this is a basic "
        "summary of the wording flagged for review:",
        "",
    ]
    for s in selected[:12]:
        lines.append(f"- “{s.text.strip()[:200]}”")
    lines.append("")
    lines.append(
        "Run the backend with the full model (GPU) to get a grounded, "
        "citation-backed explanation of the bias and framing."
    )
    return "\n".join(lines)


def fact_checks_from_record(record: dict[str, Any], selected: list[SelectedSentence]):
    claim_records = [
        claim
        for w in record.get("context_windows") or []
        for claim in (w.get("claims") or [])
    ]
    if claim_records:
        return fact_checker.fact_checks_from_v3_claims(claim_records)
    return fact_checker.conservative_fallback([s.text for s in selected])


def record_to_response(
    record: dict[str, Any],
    relevance_meta: dict[str, Any],
    *,
    mode: str,
    llm_used: bool,
    evidence_mode: str,
    evidence_unavailable_reason: str | None = None,
) -> AnalyzeResponse:
    all_predictions = record.get("sentence_predictions") or []
    selected_records = record.get("selected_sentences") or []

    signals = [
        scoring.sentence_signal_v3(p.get("p_bias"), p.get("p_opinionated_style"))
        for p in all_predictions
    ]
    score = scoring.compute_score(signals)
    level = scoring.bucket(score)
    caption = scoring.caption(
        score, len(selected_records), record.get("sentence_count", len(all_predictions))
    )

    selected_out = [selected_sentence_from_v3(p) for p in selected_records]
    report = combine_reports(record, selected_out)
    fact_checks = fact_checks_from_record(record, selected_out)

    meta: dict[str, Any] = {
        "mode": mode,
        "stack": "v3",
        "sentence_count": record.get("sentence_count", 0),
        "selected_count": len(selected_records),
        "window_count": len(record.get("context_windows") or []),
        "llm_used": llm_used,
        "llm_revision": record.get("llm_revision"),
        "relevance": relevance_meta,
        "selection_policy": record.get("selection_policy"),
        "abstention_summary": record.get("abstention_summary"),
        "evidence_enabled": evidence_mode != "off",
        "evidence_mode": evidence_mode,
        "evidence_item_count": len(record.get("evidence") or []),
    }
    if evidence_unavailable_reason:
        meta["evidence_unavailable_reason"] = evidence_unavailable_reason

    return AnalyzeResponse(
        overall_score=score,
        bias_level=level,
        score_caption=caption,
        report=report,
        fact_checks=fact_checks,
        selected_sentences=selected_out,
        meta=meta,
    )
