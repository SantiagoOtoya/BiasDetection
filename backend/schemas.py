"""Shared request/response models for the Bias Detection backend.

These define the contract between the Chrome extension and the server. The mock
pipeline (Phase 3) and the real SBERT+Llama pipeline (Phases 6-9) both return an
``AnalyzeResponse``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class FactStatus(str, Enum):
    verified = "verified"
    unverified = "unverified"
    disputed = "disputed"
    false = "false"


class BiasLevel(str, Enum):
    low = "Low"
    moderate = "Moderate"
    high = "High"
    severe = "Severe"
    unknown = "Unknown"


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Main article text extracted from the page.")
    title: Optional[str] = Field(default=None, description="Article title, if known.")
    url: Optional[str] = Field(default=None, description="Source page URL, if known.")
    lead_text: str = Field(
        default="",
        description="Topic anchor (headline + first paragraph) for the relevance filter.",
    )
    max_windows: int = Field(
        default=0,
        ge=0,
        description="Optional cap on LLM context windows. 0 means no cap.",
    )


class SelectedSentence(BaseModel):
    index: int
    text: str
    bias_label: str
    bias_probability: float
    opinion_label: str
    opinion_probability: float
    # Calibrated-selection provenance (empty when running legacy argmax).
    selection_reasons: list[str] = Field(default_factory=list)
    abstention_reasons: list[str] = Field(default_factory=list)


class FactCheck(BaseModel):
    claim: str
    status: FactStatus = FactStatus.unverified
    rationale: str = ""
    # Optional trusted-evidence citation ids (e.g. ["E1"]) when evidence-backed.
    evidence_ids: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    overall_score: float = Field(..., ge=0, le=100)
    bias_level: BiasLevel = BiasLevel.unknown
    score_caption: str = ""
    report: str = ""
    fact_checks: list[FactCheck] = Field(default_factory=list)
    selected_sentences: list[SelectedSentence] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    mode: str
    sbert_loaded: bool = False
    llm_loaded: bool = False
    detail: str = ""
