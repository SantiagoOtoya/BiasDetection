"""Analyzer abstraction for the backend.

Phase 3 ships a ``MockAnalyzer`` so the extension <-> backend round-trip can be
tested before models are wired in. Phases 6-9 add a ``RealAnalyzer`` (SBERT +
Llama + fact-checker) selected via the ``ANALYZER_MODE`` environment variable.
"""

from __future__ import annotations

import os
import re

from schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BiasLevel,
    FactCheck,
    FactStatus,
    SelectedSentence,
)


class Analyzer:
    """Base analyzer interface."""

    mode: str = "base"
    sbert_loaded: bool = False
    llm_loaded: bool = False

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:  # pragma: no cover
        raise NotImplementedError


class MockAnalyzer(Analyzer):
    """Returns deterministic placeholder output shaped like the real pipeline.

    Useful for verifying the extension UI, CORS, and the request/response
    contract without downloading or loading any models.
    """

    mode = "mock"

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        # Heuristics-only relevance pass (no embeddings in mock mode), so the
        # extension's relevance debug counts can be exercised without models.
        import relevance_filter

        raw_sentences = relevance_filter.split_into_paragraph_sentences(
            request.text, _split_sentences
        )

        relevance = relevance_filter.filter_sentences(
            raw_sentences,
            title=request.title or "",
            lead_text=request.lead_text or "",
            embed_fn=None,
            enable_semantic=False,
        )
        sentences = relevance.kept_sentences
        sentence_count = len(sentences)

        # Pick a few "biased" sentences deterministically for demonstration.
        selected: list[SelectedSentence] = []
        for index, sentence in enumerate(sentences):
            if index % 4 == 1 and len(sentence) > 30:
                selected.append(
                    SelectedSentence(
                        index=index,
                        text=sentence,
                        bias_label="Biased",
                        bias_probability=0.82,
                        opinion_label="Expresses writer's opinion",
                        opinion_probability=0.71,
                    )
                )

        selected_count = len(selected)
        score = min(100.0, selected_count / max(1, sentence_count) * 220.0)
        level = _bucket(score)

        report = _mock_report(request.title, selected)
        fact_checks = _mock_fact_checks(selected)

        return AnalyzeResponse(
            overall_score=round(score, 1),
            bias_level=level,
            score_caption=f"{selected_count} of {sentence_count} sentences flagged (mock).",
            report=report,
            fact_checks=fact_checks,
            selected_sentences=selected,
            meta={
                "mode": self.mode,
                "sentence_count": sentence_count,
                "selected_count": selected_count,
                "relevance": relevance.to_meta(len(raw_sentences)),
                "note": "Mock output. Real SBERT/Llama analysis is added in later phases.",
            },
        )


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", text)
    return [p.strip() for p in pieces if p.strip()] or [text]


def _bucket(score: float) -> BiasLevel:
    if score < 20:
        return BiasLevel.low
    if score < 45:
        return BiasLevel.moderate
    if score < 70:
        return BiasLevel.high
    return BiasLevel.severe


def _mock_report(title: str | None, selected: list[SelectedSentence]) -> str:
    heading = title.strip() if title else "this page"
    if not selected:
        return (
            f"This is placeholder output for {heading}. No strongly biased or "
            "opinionated wording was flagged by the mock analyzer. The real "
            "analysis pipeline will replace this text."
        )
    first = selected[0].text
    return (
        f"This is placeholder output for {heading}. The mock analyzer flagged "
        f"{len(selected)} passage(s) that would receive focused analysis.\n\n"
        f"For example, the wording in “{first[:160]}” would be examined for "
        "loaded language and framing. Once the SBERT classifier and the media-bias "
        "report model are connected, this section will contain a grounded, "
        "chronological explanation of the biased or opinionated wording found on the page."
    )


def _mock_fact_checks(selected: list[SelectedSentence]) -> list[FactCheck]:
    statuses = [FactStatus.unverified, FactStatus.disputed, FactStatus.verified]
    checks: list[FactCheck] = []
    for i, sentence in enumerate(selected[:3]):
        checks.append(
            FactCheck(
                claim=sentence.text[:180],
                status=statuses[i % len(statuses)],
                rationale="Mock status. The conservative fact-checker component fills this in later.",
            )
        )
    return checks


def build_analyzer(mode: str) -> Analyzer:
    """Factory.

    Modes:
      - "mock":        deterministic placeholder, no models (default).
      - "gpu":         real SBERT + Llama (requires CUDA).
      - "cpu":         real SBERT + Llama on CPU (slow; for local testing).
      - "prompt-only": real SBERT + scoring + fact-check fallback, no Llama.

    Real modes import torch/sentence-transformers lazily so the mock server can
    run without the ML stack installed. If a real analyzer fails to construct,
    the error is raised (we do not silently fall back to mock for real modes).
    """
    normalized = (mode or "mock").strip().lower()
    if normalized == "mock":
        return MockAnalyzer()

    if normalized in ("gpu", "cpu", "prompt-only", "prompt_only"):
        from real_pipeline import RealAnalyzer

        canonical = "prompt-only" if normalized in ("prompt-only", "prompt_only") else normalized
        return RealAnalyzer(
            mode=canonical,
            selection_mode=_env_selection_mode(),
            enable_evidence=_env_flag("ENABLE_EVIDENCE", default=False),
            evidence_provider=os.environ.get("EVIDENCE_PROVIDER", "brave"),
            max_evidence_items=_env_int("MAX_EVIDENCE_ITEMS", 5),
            evidence_timeout_seconds=_env_float("EVIDENCE_TIMEOUT_SECONDS", 10.0),
            enable_relevance=_env_flag("ENABLE_RELEVANCE", default=True),
            enable_semantic_relevance=_env_flag("ENABLE_SEMANTIC_RELEVANCE", default=True),
            relevance_threshold=_env_float("RELEVANCE_THRESHOLD", 0.18),
        )

    # Unknown mode: keep the server usable rather than failing to start.
    return MockAnalyzer()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_selection_mode() -> str:
    mode = os.environ.get("SELECTION_MODE", "auto").strip().lower()
    if mode in ("auto", "argmax", "calibrated"):
        return mode
    return "auto"
