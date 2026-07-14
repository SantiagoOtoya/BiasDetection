#!/usr/bin/env python
"""Run SBERT bias/opinion sentence detection and hand context to Llama.

The fine-tuned SentenceTransformer and its two MLP classification heads are
loaded from the saved training output. Sentence embeddings are used internally
for classification; only selected sentence text plus surrounding article
context is sent to the LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for import_path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import evidence_retrieval
import evidence_assessment
import runtime_config
import artifact_manifest
import data_pipeline
import finetune_all_mpnet_babe as training


LOGGER = logging.getLogger("infer_bias_llm")
DEFAULT_LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_LLM_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
CALIBRATION_SCHEMA_VERSION = "calibration/v3"
LEGACY_CALIBRATION_SCHEMA_VERSIONS = {
    1,
    2,
    "calibration/v1",
    "calibration/legacy-v1",
}
DECISION_POLICY_VERSION = "classifier_decision/v1"
MEDIA_BIAS_SYSTEM_PROMPT = """You write a restrained, user-facing media-bias report.

Analyze only sentences selected by their bias_assessment and use surrounding context
only to interpret their wording. opinion_style is auxiliary writing-style context: it
does not establish bias, truth, support, contradiction, or report eligibility.

Discuss concrete framing, loaded wording, omissions visible in the supplied passage,
and neutral alternatives in chronological order. A possible_bias item is uncertain
and must never be described as a confident bias finding.

Evidence rules are strict. Only the supplied claim evidence_status may establish
supported or contradicted. Retrieval scores, semantic similarity, source reputation,
search snippets, article repetition, and classifier output are not factual proof.
Cite only citation_ids attached to a claim assessment. If evidence is insufficient,
not verifiable, not assessed, or retrieval failed, say only that the supplied material
does not establish the claim. Never invent facts, corrections, citations, or URLs.

Do not mention models, classifiers, prompts, systems, or internal processing. Write
cohesive prose, keep quotes short, avoid speculation about motives, and return only
the finished report text."""


@dataclass
class ArticleRecord:
    article_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class SentencePrediction:
    index: int
    text: str
    bias_label: str
    bias_probability: float
    bias_probabilities: dict[str, float]
    opinion_label: str
    opinion_probability: float
    opinion_probabilities: dict[str, float]
    selected: bool
    selection_reasons: list[str] | None = None
    abstention_reasons: list[str] | None = None
    bias_selection_score: float | None = None
    opinion_selection_score: float | None = None
    bias_calibrated_probability: float | None = None
    bias_calibrated_probabilities: dict[str, float] | None = None
    opinion_calibrated_probability: float | None = None
    opinion_calibrated_probabilities: dict[str, float] | None = None
    bias_calibrated_label: str | None = None
    opinion_calibrated_label: str | None = None
    bias_positive: bool | None = None
    report_trigger: bool | None = None
    report_trigger_score: float | None = None
    uncertainty_status: str = "uncertain"
    bias_assessment: str | None = None
    opinion_style: str | None = None
    p_bias: float | None = None
    p_opinionated_style: float | None = None
    bias_logit: float | None = None
    opinionated_style_logit: float | None = None
    legacy_compatibility: bool = True
    context_text: str = ""
    model_input: str = ""
    evidence_status: str = "not_assessed"
    claims: list[dict[str, Any]] | None = None


@dataclass
class ContextWindow:
    start_index: int
    end_index: int
    selected_sentence_indices: list[int]
    text: str


@dataclass(frozen=True)
class SentenceContextInput:
    index: int
    text: str
    context_before: str
    context_after: str
    context_text: str
    model_input: str


@dataclass(frozen=True)
class CalibrationHeadConfig:
    enabled: bool
    threshold: float | None
    temperature: float
    target_precision: float | None = None
    validation_precision: float | None = None
    validation_recall: float | None = None
    validation_selected_count: int | None = None
    validation_coverage: float | None = None
    validation_candidate_count: int | None = None
    validation_candidate_coverage: float | None = None
    validation_positive_support: int | None = None
    validation_precision_wilson_lower_bound: float | None = None
    precision_confidence: float | None = None
    coverage_unit: str | None = None
    disabled_reason: str | None = None


@dataclass(frozen=True)
class ProbabilityCalibratorConfig:
    method: str
    input_kind: str
    temperature: float


@dataclass(frozen=True)
class ThresholdRegionConfig:
    enabled: bool
    threshold: float | None
    metric: str
    target: float
    confidence: float
    minimum_predicted_support: int
    selected_support: int
    observed_metric: float | None
    wilson_lower_bound: float
    coverage: float
    disabled_reason: str | None = None


@dataclass(frozen=True)
class DecisionHeadConfig:
    calibrator: ProbabilityCalibratorConfig
    lower: ThresholdRegionConfig
    upper: ThresholdRegionConfig


@dataclass(frozen=True)
class SelectionConfig:
    requested_mode: str
    effective_mode: str
    calibration_file: Path | None = None
    artifact_manifest_file: Path | None = None
    calibration_reason: str | None = None
    artifact_binding: dict[str, Any] | None = None
    legacy_compatibility: bool = False
    bias: CalibrationHeadConfig | None = None
    opinion: CalibrationHeadConfig | None = None
    report_trigger: CalibrationHeadConfig | None = None
    bias_decision: DecisionHeadConfig | None = None
    opinion_style_decision: DecisionHeadConfig | None = None
    include_possible_bias: bool = False


@dataclass(frozen=True)
class SelectionOutcome:
    selected: bool
    selection_reasons: list[str]
    abstention_reasons: list[str]
    bias_selection_score: float | None
    opinion_selection_score: float | None
    report_trigger_score: float | None
    bias_positive: bool
    report_trigger: bool
    uncertainty_status: str
    bias_assessment: str | None = None
    opinion_style: str | None = None


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_sbert_dir = project_root / "models" / "all-mpnet-base-v2-babe-v3" / "best"
    default_output = script_dir / "outputs" / "bias_llm_results.jsonl"

    parser = argparse.ArgumentParser(
        description=(
            "Classify article sentences with the fine-tuned SBERT bias/opinion "
            "heads, then send selected sentence context to Llama 3.1 8B."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--article-text", help="Raw article text to analyze.")
    input_group.add_argument(
        "--article-file",
        type=Path,
        help="UTF-8 text file containing one article.",
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="CSV or parquet file containing article rows.",
    )

    parser.add_argument(
        "--article-column",
        default="article",
        help=(
            "Article text column for --input-file. If left as the default and "
            "'article' is absent, the script falls back to 'text'."
        ),
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Optional article identifier column for --input-file.",
    )
    parser.add_argument(
        "--csv-sep",
        default=None,
        help="CSV separator. Omit to let pandas sniff the delimiter.",
    )
    parser.add_argument(
        "--sbert-model-dir",
        type=Path,
        default=default_sbert_dir,
        help="Fine-tuned SentenceTransformer checkpoint directory.",
    )
    parser.add_argument(
        "--classification-heads",
        type=Path,
        default=None,
        help="Path to classification_heads.pt. Defaults to SBERT dir/classification_heads.pt.",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="Hugging Face causal LM model id for generation.",
    )
    parser.add_argument(
        "--llm-revision",
        default=DEFAULT_LLM_REVISION,
        help="Pinned Hugging Face model revision used for tokenizer and model loading.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=default_output,
        help="JSONL output file. Use '-' to write JSONL records to stdout.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for SBERT classification. Defaults to cuda if available, else cpu.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--context-sentences",
        type=int,
        default=2,
        help="Number of sentences before and after each selected sentence.",
    )
    parser.add_argument(
        "--max-windows-per-article",
        type=int,
        default=0,
        help="Optional cap on LLM context windows per article. 0 means no cap.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["auto", "calibrated", "legacy", "argmax"],
        default="calibrated",
        help=(
            "Sentence selection policy. 'calibrated' and 'auto' require a "
            "promoted manifest-bound calibration. 'legacy' explicitly permits "
            "an unbound compatibility calibration; 'argmax' is explicit legacy "
            "argmax behavior."
        ),
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=None,
        help="Optional calibration.json path. Defaults to sbert-model-dir/calibration.json.",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=None,
        help=(
            "Optional artifact_manifest.json path. Defaults to "
            "sbert-model-dir/artifact_manifest.json for strict calibration."
        ),
    )
    parser.add_argument(
        "--bias-threshold",
        type=float,
        default=None,
        help="Legacy-only override for calibrated P(Biased) threshold.",
    )
    parser.add_argument(
        "--opinion-threshold",
        type=float,
        default=None,
        help=(
            "Legacy-only override for calibrated non-factual opinion threshold, defined as "
            "1 - P(Entirely factual)."
        ),
    )
    parser.add_argument(
        "--include-all-sentence-predictions",
        action="store_true",
        help="Include selected and abstained sentence predictions in JSONL output.",
    )
    parser.add_argument(
        "--enable-evidence",
        action="store_true",
        help=(
            "Deprecated compatibility alias for --evidence-mode web."
        ),
    )
    parser.add_argument(
        "--include-possible-bias",
        action="store_true",
        help="Also select possible_bias candidates; clear_bias is selected by default.",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=["off", "web", "corpus", "hybrid"],
        default=None,
        help=(
            "Trusted evidence mode. Defaults to off; --enable-evidence remains "
            "a compatibility alias for web. Corpus settings use the existing "
            "QDRANT_* environment contract."
        ),
    )
    parser.add_argument(
        "--evidence-provider",
        default="brave",
        choices=["brave"],
        help="Trusted evidence search provider. Brave requires BRAVE_SEARCH_API_KEY.",
    )
    parser.add_argument(
        "--max-evidence-items",
        type=int,
        default=5,
        help="Maximum trusted evidence items to attach to each context window.",
    )
    parser.add_argument(
        "--evidence-timeout-seconds",
        type=float,
        default=10.0,
        help="Timeout for each evidence search request.",
    )
    parser.add_argument(
        "--trusted-sources-file",
        type=Path,
        default=None,
        help="Optional JSON file overriding the default trusted-source registry.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Explicit dotenv file for Brave/Qdrant settings. Existing process "
            "environment variables take precedence; no env file is loaded implicitly."
        ),
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Build prompts and output records without loading or calling the LLM.",
    )
    parser.add_argument(
        "--allow-cpu-llm",
        action="store_true",
        help="Allow local LLM generation on CPU. This is usually very slow for 8B models.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN from the environment/cache.",
    )
    args = parser.parse_args()
    try:
        args.evidence_mode = resolve_evidence_mode(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def resolve_evidence_mode(args: argparse.Namespace) -> str:
    """Resolve the new explicit mode while preserving legacy caller namespaces."""

    requested_mode = getattr(args, "evidence_mode", None)
    legacy_enabled = bool(getattr(args, "enable_evidence", False))
    if requested_mode is None:
        return "web" if legacy_enabled else "off"

    mode = str(requested_mode).casefold()
    if mode not in {"off", "web", "corpus", "hybrid"}:
        raise ValueError(f"Unsupported evidence mode: {requested_mode}")
    if legacy_enabled and mode != "web":
        raise ValueError("--enable-evidence is only compatible with --evidence-mode web")
    return mode


def require_inference_dependencies(prompt_only: bool) -> None:
    training.import_training_dependencies()
    if prompt_only:
        return

    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing inference package: transformers\n"
            "Install inference dependencies with:\n"
            "  python -m pip install -r requirements-inference.txt"
        ) from exc


def load_article_records(args: argparse.Namespace) -> list[ArticleRecord]:
    if args.article_text is not None:
        return [
            ArticleRecord(
                article_id="article-0",
                text=args.article_text,
                metadata={"source": "article-text"},
            )
        ]

    if args.article_file is not None:
        text = args.article_file.read_text(encoding="utf-8")
        return [
            ArticleRecord(
                article_id=args.article_file.stem,
                text=text,
                metadata={"source": str(args.article_file)},
            )
        ]

    return load_article_records_from_table(args)


def load_article_records_from_table(args: argparse.Namespace) -> list[ArticleRecord]:
    pd = training.pd
    assert pd is not None

    input_file: Path = args.input_file
    suffix = input_file.suffix.casefold()
    if suffix == ".parquet":
        df = pd.read_parquet(input_file)
    elif suffix == ".csv":
        if args.csv_sep is None:
            df = pd.read_csv(input_file, sep=None, engine="python", dtype=str, keep_default_na=False)
        else:
            df = pd.read_csv(input_file, sep=args.csv_sep, dtype=str, keep_default_na=False)
    else:
        raise ValueError(f"Unsupported input file suffix: {input_file.suffix}")

    article_column = resolve_article_column(df, args.article_column)
    id_column = resolve_id_column(df, args.id_column)
    records: list[ArticleRecord] = []
    for row_index, row in df.fillna("").iterrows():
        text = str(row.get(article_column, "")).strip()
        if not text:
            continue

        if id_column is not None:
            article_id = str(row.get(id_column, "")).strip() or f"row-{row_index}"
        else:
            article_id = f"row-{row_index}"

        metadata = {
            "source": str(input_file),
            "row_index": int(row_index),
            "article_column": article_column,
        }
        for column in ["news_link", "outlet", "topic", "type"]:
            if column in df.columns:
                metadata[column] = str(row.get(column, ""))
        records.append(ArticleRecord(article_id=article_id, text=text, metadata=metadata))

    return records


def resolve_article_column(df: Any, requested_column: str) -> str:
    if requested_column in df.columns:
        return requested_column
    if requested_column == "article" and "text" in df.columns:
        LOGGER.warning("Column 'article' not found; falling back to 'text'.")
        return "text"
    raise ValueError(
        f"Article column '{requested_column}' not found. "
        f"Available columns: {', '.join(df.columns)}"
    )


def resolve_id_column(df: Any, requested_column: str | None) -> str | None:
    if requested_column:
        if requested_column not in df.columns:
            raise ValueError(
                f"ID column '{requested_column}' not found. "
                f"Available columns: {', '.join(df.columns)}"
            )
        return requested_column

    for candidate in ["uuid", "news_link", "source_file"]:
        if candidate in df.columns:
            return candidate
    return None


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    pieces = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", text)
    sentences = [piece.strip() for piece in pieces if piece.strip()]
    return sentences or [text]


def build_sentence_context_inputs(
    sentences: list[str],
    context_sentences: int,
) -> list[SentenceContextInput]:
    """Build context records while keeping classifier input target-only."""

    if context_sentences < 0:
        raise ValueError("context_sentences must be non-negative")
    inputs: list[SentenceContextInput] = []
    for index, sentence in enumerate(sentences):
        before = " ".join(sentences[max(0, index - context_sentences) : index])
        after = " ".join(sentences[index + 1 : index + context_sentences + 1])
        context_text = " ".join(part for part in (before, sentence, after) if part)
        model_input = data_pipeline.serialize_classifier_input(sentence)
        inputs.append(
            SentenceContextInput(
                index=index,
                text=sentence,
                context_before=before,
                context_after=after,
                context_text=context_text,
                model_input=model_input,
            )
        )
    return inputs


def _normalize_sentence_inputs(
    values: list[str] | list[SentenceContextInput],
) -> list[SentenceContextInput]:
    if not values:
        return []
    if isinstance(values[0], SentenceContextInput):
        return list(values)  # type: ignore[arg-type]
    return build_sentence_context_inputs([str(value) for value in values], 0)


def load_sbert_and_heads(
    model_dir: Path,
    heads_path: Path | None,
    device: Any,
    *,
    compatibility_mode: str = "strict",
) -> tuple[Any, Any, Any, dict[str, int], dict[str, int]]:
    torch = training.torch
    SentenceTransformer = training.SentenceTransformer
    assert torch is not None
    assert SentenceTransformer is not None

    heads_path = heads_path or (model_dir / "classification_heads.pt")
    if not heads_path.exists():
        raise FileNotFoundError(f"Classification heads checkpoint not found: {heads_path}")

    LOGGER.info("Loading SBERT checkpoint from %s", model_dir)
    model = SentenceTransformer(str(model_dir), device=str(device))
    metadata_path = model_dir / "training_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        max_seq_length = metadata.get("max_seq_length")
        if max_seq_length:
            model.max_seq_length = int(max_seq_length)
    model.eval()

    checkpoint = torch.load(heads_path, map_location=device)
    try:
        bias_head, opinion_head, classifier_mode = (
            training.load_classification_heads_from_checkpoint(
                checkpoint, device, compatibility_mode=compatibility_mode
            )
        )
    except training.CheckpointCompatibilityError as exc:
        raise CalibrationValidationError(str(exc)) from exc
    bias_label2id = {
        str(label): int(index) for label, index in checkpoint["bias_label2id"].items()
    }
    if classifier_mode == training.PRODUCTION_CLASSIFIER_MODE:
        opinion_label2id = {
            str(label): int(index)
            for label, index in checkpoint["opinion_style_label2id"].items()
        }
        bias_head.head_type = checkpoint["bias_head_type"]
        opinion_head.head_type = checkpoint["opinion_style_head_type"]
        opinion_head.opinion_style_target_mapping = dict(
            checkpoint["opinion_style_target_mapping"]
        )
        for head in (bias_head, opinion_head):
            head.classifier_input_contract = checkpoint["classifier_input_contract"]
    else:
        opinion_label2id = {
            str(label): int(index)
            for label, index in checkpoint["opinion_label2id"].items()
        }
    for head in (bias_head, opinion_head):
        head.checkpoint_schema_version = checkpoint.get("checkpoint_schema_version")
        head.classifier_mode = classifier_mode
    bias_head.checkpoint_declared_head_type = checkpoint.get("bias_head_type")
    opinion_head.checkpoint_declared_head_type = (
        checkpoint.get("opinion_style_head_type")
        if classifier_mode == training.PRODUCTION_CLASSIFIER_MODE
        else checkpoint.get("head_type")
    )
    opinion_head.classifier_head_type = checkpoint.get("classifier_head_type")
    bias_head.eval()
    opinion_head.eval()

    return model, bias_head, opinion_head, bias_label2id, opinion_label2id


class CalibrationValidationError(ValueError):
    """Raised when a calibration sidecar cannot be safely applied."""


def legacy_argmax_selection_config() -> SelectionConfig:
    """Return the explicit compatibility configuration for raw argmax selection."""

    return SelectionConfig(
        requested_mode="argmax",
        effective_mode="argmax",
        calibration_reason="legacy_argmax_requested",
        legacy_compatibility=True,
    )


def validate_probability_threshold(name: str, value: float | None) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")


def calibration_head_from_json(
    head_name: str,
    data: dict[str, Any] | None,
    threshold_override: float | None,
    *,
    strict: bool = False,
) -> CalibrationHeadConfig:
    validate_probability_threshold(f"--{head_name}-threshold", threshold_override)
    if data is None:
        if strict:
            raise CalibrationValidationError(
                f"Strict calibration is missing the {head_name} head configuration."
            )
        if threshold_override is None:
            return CalibrationHeadConfig(
                enabled=False,
                threshold=None,
                temperature=1.0,
                disabled_reason="missing_calibration_head",
            )
        return CalibrationHeadConfig(
            enabled=True,
            threshold=threshold_override,
            temperature=1.0,
        )

    if strict:
        required = {
            "enabled",
            "threshold",
            "temperature",
            "target_precision",
            "validation_precision",
            "validation_precision_wilson_lower_bound",
            "validation_recall",
            "validation_selected_count",
            "validation_coverage",
            "validation_candidate_count",
            "validation_candidate_coverage",
            "validation_positive_support",
            "precision_confidence",
            "disabled_reason",
            "coverage_unit",
        }
        missing = required.difference(data)
        if missing:
            raise CalibrationValidationError(
                f"Strict calibration {head_name} head is missing fields: {sorted(missing)}"
            )
    temperature = float(data["temperature"] if strict else data.get("temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError(f"Calibration temperature for {head_name} must be > 0.")

    threshold_value = data["threshold"] if strict else data.get("threshold")
    threshold = float(threshold_value) if threshold_value is not None else None
    if threshold_override is not None:
        threshold = threshold_override
    validate_probability_threshold(f"{head_name} threshold", threshold)

    enabled = bool(data["enabled"] if strict else data.get("enabled", threshold is not None))
    disabled_reason = data.get("disabled_reason")
    if threshold_override is not None:
        enabled = True
        disabled_reason = None
    if threshold is None:
        enabled = False
        disabled_reason = disabled_reason or "missing_threshold"

    return CalibrationHeadConfig(
        enabled=enabled,
        threshold=threshold,
        temperature=temperature,
        target_precision=_optional_float(data.get("target_precision")),
        validation_precision=_optional_float(data.get("validation_precision")),
        validation_recall=_optional_float(data.get("validation_recall")),
        validation_selected_count=_optional_int(data.get("validation_selected_count")),
        validation_coverage=_optional_float(data.get("validation_coverage")),
        validation_candidate_count=_optional_int(data.get("validation_candidate_count")),
        validation_candidate_coverage=_optional_float(
            data.get("validation_candidate_coverage")
        ),
        validation_positive_support=_optional_int(data.get("validation_positive_support")),
        validation_precision_wilson_lower_bound=_optional_float(
            data.get("validation_precision_wilson_lower_bound")
        ),
        precision_confidence=_optional_float(data.get("precision_confidence")),
        coverage_unit=(
            str(data["coverage_unit"])
            if strict or data.get("coverage_unit") is not None
            else None
        ),
        disabled_reason=disabled_reason,
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _label_mapping(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise CalibrationValidationError(f"{name} must be a label-to-ID mapping.")
    try:
        return {str(label): int(index) for label, index in value.items()}
    except (TypeError, ValueError) as exc:
        raise CalibrationValidationError(f"{name} contains an invalid label ID.") from exc


def _artifact_entry_path(entry: Any, *, name: str) -> Path:
    if not isinstance(entry, dict) or "path" not in entry:
        raise CalibrationValidationError(f"Artifact manifest is missing {name}.")
    path = (PROJECT_ROOT / str(entry["path"])).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise CalibrationValidationError(
            f"Artifact manifest {name} path escapes the project root."
        ) from exc
    return path


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationValidationError(f"Could not read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise CalibrationValidationError(f"{description} must be a JSON object.")
    return value


def _strict_calibrator_from_json(name: str, data: Any) -> ProbabilityCalibratorConfig:
    if not isinstance(data, dict):
        raise CalibrationValidationError(f"Strict calibration is missing {name} calibrator.")
    if data.get("schema_version") != "binary_calibrator/v1":
        raise CalibrationValidationError(f"Unsupported {name} calibrator schema.")
    if data.get("method") != "temperature_scaling" or data.get("input_kind") != "binary_logit":
        raise CalibrationValidationError(
            f"{name} calibrator must consume raw binary logits with temperature scaling."
        )
    try:
        temperature = float(data["parameters"]["temperature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationValidationError(f"{name} calibrator has invalid parameters.") from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise CalibrationValidationError(f"{name} temperature must be finite and positive.")
    return ProbabilityCalibratorConfig(
        method="temperature_scaling", input_kind="binary_logit", temperature=temperature
    )


def _strict_region_from_json(name: str, data: Any) -> ThresholdRegionConfig:
    if not isinstance(data, dict):
        raise CalibrationValidationError(f"Strict calibration is missing {name}.")
    required = {
        "enabled", "disabled_reason", "threshold", "metric", "target",
        "confidence", "minimum_predicted_support", "selected_support",
        "observed_metric", "wilson_lower_bound", "coverage",
    }
    missing = required.difference(data)
    if missing:
        raise CalibrationValidationError(f"{name} is missing fields: {sorted(missing)}")
    threshold = _optional_float(data["threshold"])
    enabled = bool(data["enabled"])
    if enabled and threshold is None:
        raise CalibrationValidationError(f"Enabled region {name} has no threshold.")
    if not enabled and threshold is not None:
        raise CalibrationValidationError(f"Disabled region {name} must have threshold=null.")
    validate_probability_threshold(name, threshold)
    minimum = int(data["minimum_predicted_support"])
    selected = int(data["selected_support"])
    lcb = float(data["wilson_lower_bound"])
    target = float(data["target"])
    if minimum <= 0 or selected < 0 or not 0.0 <= lcb <= 1.0 or not 0.0 <= target <= 1.0:
        raise CalibrationValidationError(f"Region {name} has invalid support or metric values.")
    if enabled and (selected < minimum or lcb < target):
        raise CalibrationValidationError(f"Enabled region {name} does not satisfy its gate.")
    if not enabled and not data.get("disabled_reason"):
        raise CalibrationValidationError(f"Disabled region {name} requires a reason.")
    return ThresholdRegionConfig(
        enabled=enabled,
        threshold=threshold,
        metric=str(data["metric"]),
        target=target,
        confidence=float(data["confidence"]),
        minimum_predicted_support=minimum,
        selected_support=selected,
        observed_metric=_optional_float(data["observed_metric"]),
        wilson_lower_bound=lcb,
        coverage=float(data["coverage"]),
        disabled_reason=(str(data["disabled_reason"]) if data.get("disabled_reason") else None),
    )


def _strict_decision_head_from_json(
    name: str, calibrator_data: Any, lower_name: str, upper_name: str,
    thresholds: Mapping[str, Any],
) -> DecisionHeadConfig:
    calibrator = _strict_calibrator_from_json(name, calibrator_data)
    lower = _strict_region_from_json(lower_name, thresholds.get(lower_name))
    upper = _strict_region_from_json(upper_name, thresholds.get(upper_name))
    if lower.enabled and upper.enabled and float(lower.threshold) >= float(upper.threshold):
        raise CalibrationValidationError(f"{name} thresholds are not strictly ordered.")
    return DecisionHeadConfig(calibrator=calibrator, lower=lower, upper=upper)


def _validate_strict_calibration_schema(data: dict[str, Any]) -> None:
    if data.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise CalibrationValidationError(
            "Strict inference requires calibration/v3; use --selection-mode legacy "
            "only for an explicitly requested compatibility sidecar."
        )
    required = {
        "artifact_binding", "calibration_data", "head_types", "label_mappings",
        "calibrators", "thresholds", "decision_policy_version", "calibration_status",
    }
    missing = required.difference(data)
    if missing:
        raise CalibrationValidationError(f"Strict calibration is missing fields: {sorted(missing)}")
    if data["decision_policy_version"] != DECISION_POLICY_VERSION:
        raise CalibrationValidationError("Unsupported classifier decision policy version.")
    if data["calibration_status"] != "valid":
        raise CalibrationValidationError("Strict inference requires a valid calibration artifact.")
    if "heads" in data or "report_trigger" in data:
        raise CalibrationValidationError(
            "calibration/v3 cannot contain legacy head or report-trigger calibration."
        )
    if set(data["calibrators"]) != {"bias", "opinion_style"}:
        raise CalibrationValidationError("Strict calibration must contain exactly two calibrators.")
    if set(data["thresholds"]) != {
        "bias_no_clear_max", "bias_clear_min",
        "opinion_objective_max", "opinion_opinionated_min",
    }:
        raise CalibrationValidationError("Strict calibration must contain exactly four regions.")
    _strict_decision_head_from_json(
        "bias", data["calibrators"]["bias"], "bias_no_clear_max", "bias_clear_min",
        data["thresholds"],
    )
    _strict_decision_head_from_json(
        "opinion_style", data["calibrators"]["opinion_style"],
        "opinion_objective_max", "opinion_opinionated_min", data["thresholds"],
    )
    calibration_data = data["calibration_data"]
    if not isinstance(calibration_data, dict):
        raise CalibrationValidationError("Strict calibration data binding must be an object.")
    data_required = {
        "schema_version",
        "partition_id",
        "record_count",
        "records_sha256",
        "split_partition_assignment_sha256",
    }
    missing_data = data_required.difference(calibration_data)
    if missing_data:
        raise CalibrationValidationError(
            "Strict calibration data binding is missing fields: "
            f"{sorted(missing_data)}"
        )
    if calibration_data["schema_version"] != "calibration_partition/v1":
        raise CalibrationValidationError("Unsupported calibration data binding schema.")
    binding = data["artifact_binding"]
    if not isinstance(binding, dict):
        raise CalibrationValidationError("Strict calibration artifact_binding must be an object.")
    binding_required = {
        "artifact_manifest_sha256",
        "model_sha256",
        "heads_sha256",
        "model_artifact_path",
        "classification_heads_path",
        "split_manifest_sha256",
        "split_manifest_file_sha256",
        "canonical_data_manifest_sha256",
        "canonical_data_manifest_file_sha256",
        "calibration_partition_id",
        "calibration_records_sha256",
        "calibration_partition_assignment_sha256",
        "artifact_code_sha256",
        "calibration_code_sha256",
        "checkpoint_schema_version",
        "calibration_schema_version",
        "decision_policy_version",
        "classifier_input_contract",
    }
    missing_binding = binding_required.difference(binding)
    if missing_binding:
        raise CalibrationValidationError(
            "Strict calibration artifact binding is missing fields: "
            f"{sorted(missing_binding)}"
        )


def _validate_strict_artifact_binding(
    *,
    calibration: dict[str, Any],
    calibration_path: Path,
    artifact_manifest_path: Path,
    model_dir: Path,
    heads_path: Path,
    bias_label2id: dict[str, int],
    opinion_label2id: dict[str, int],
    bias_head_type: str,
    opinion_style_head_type: str,
    classifier_head_type: str,
    checkpoint_schema_version: str | None,
    opinion_style_target_mapping: dict[str, str],
    classifier_input_contract: str,
    allowed_promotion_states: frozenset[str] = frozenset({"eligible"}),
) -> dict[str, Any]:
    _validate_strict_calibration_schema(calibration)
    if checkpoint_schema_version != training.CLASSIFICATION_HEADS_SCHEMA_VERSION:
        raise CalibrationValidationError(
            "Strict inference requires classification_heads/v3."
        )
    if classifier_input_contract != data_pipeline.CLASSIFIER_INPUT_CONTRACT:
        raise CalibrationValidationError(
            "Strict inference requires the target-only classifier input contract."
        )
    try:
        artifact = artifact_manifest.load_artifact_manifest(
            artifact_manifest_path,
            artifact_root=PROJECT_ROOT,
            verify_files=True,
        )
    except (OSError, json.JSONDecodeError, artifact_manifest.ArtifactManifestValidationError) as exc:
        raise CalibrationValidationError(
            f"Artifact manifest validation failed: {artifact_manifest_path}"
        ) from exc
    if artifact.get("promotion_state") not in allowed_promotion_states:
        raise CalibrationValidationError(
            "Artifact promotion state is not allowed for this strict operation."
        )
    try:
        core = artifact["core"]
        files = core["files"]
    except (KeyError, TypeError) as exc:
        raise CalibrationValidationError("Artifact manifest has no valid core files.") from exc
    model_entry = files.get("model")
    heads_entry = files.get("classification_heads")
    split_entry = files.get("split_manifest")
    canonical_entry = files.get("canonical_data_manifest")
    if _artifact_entry_path(model_entry, name="model") != model_dir.resolve():
        raise CalibrationValidationError(
            "Artifact manifest model path does not match the loaded model directory."
        )
    if _artifact_entry_path(heads_entry, name="classification heads") != heads_path.resolve():
        raise CalibrationValidationError(
            "Artifact manifest classification heads path does not match the loaded heads."
        )
    artifact_bias_labels = _label_mapping(
        core.get("label_mappings", {}).get("bias_label2id"),
        "Artifact bias_label2id",
    )
    artifact_opinion_labels = _label_mapping(
        core.get("label_mappings", {}).get("opinion_style_label2id"),
        "Artifact opinion_style_label2id",
    )
    calibration_bias_labels = _label_mapping(
        calibration["label_mappings"].get("bias_label2id"),
        "Calibration bias_label2id",
    )
    calibration_opinion_labels = _label_mapping(
        calibration["label_mappings"].get("opinion_style_label2id"),
        "Calibration opinion_style_label2id",
    )
    if artifact_bias_labels != bias_label2id or calibration_bias_labels != bias_label2id:
        raise CalibrationValidationError(
            "Calibration or artifact bias label mapping does not match the checkpoint."
        )
    if (
        artifact_opinion_labels != opinion_label2id
        or calibration_opinion_labels != opinion_label2id
    ):
        raise CalibrationValidationError(
            "Calibration or artifact opinion label mapping does not match the checkpoint."
        )
    expected_head_types = {
        "classifier": classifier_head_type,
        "bias": bias_head_type,
        "opinion_style": opinion_style_head_type,
    }
    if core.get("head_type") != classifier_head_type or calibration.get(
        "head_types"
    ) != expected_head_types:
        raise CalibrationValidationError(
            "Calibration or artifact head types do not match the checkpoint."
        )
    calibrated_target_mapping = calibration["label_mappings"].get(
        "opinion_style_target_mapping"
    )
    if calibrated_target_mapping != opinion_style_target_mapping:
        raise CalibrationValidationError(
            "Calibration opinion-style target mapping does not match the checkpoint."
        )
    if calibration.get("classifier_input_contract") != classifier_input_contract:
        raise CalibrationValidationError(
            "Calibration classifier input contract does not match the checkpoint."
        )
    calibration_entry = artifact.get("calibration")
    if _artifact_entry_path(calibration_entry, name="calibration") != calibration_path.resolve():
        raise CalibrationValidationError(
            "Artifact manifest calibration path does not match --calibration-file."
        )
    binding = calibration["artifact_binding"]
    expected_binding = {
        "artifact_manifest_sha256": artifact.get("artifact_manifest_sha256"),
        "model_sha256": model_entry.get("sha256") if isinstance(model_entry, dict) else None,
        "heads_sha256": heads_entry.get("sha256") if isinstance(heads_entry, dict) else None,
        "model_artifact_path": model_entry.get("path") if isinstance(model_entry, dict) else None,
        "classification_heads_path": heads_entry.get("path") if isinstance(heads_entry, dict) else None,
        "split_manifest_file_sha256": split_entry.get("sha256") if isinstance(split_entry, dict) else None,
        "canonical_data_manifest_file_sha256": (
            canonical_entry.get("sha256") if isinstance(canonical_entry, dict) else None
        ),
        "artifact_code_sha256": artifact_manifest.canonical_json_sha256(
            files.get("code", [])
        ),
        "calibration_code_sha256": artifact_manifest.sha256_path(
            PROJECT_ROOT / "calibrate_sbert_heads.py"
        ),
        "checkpoint_schema_version": training.CLASSIFICATION_HEADS_SCHEMA_VERSION,
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "decision_policy_version": DECISION_POLICY_VERSION,
        "classifier_input_contract": classifier_input_contract,
    }
    for key, expected in expected_binding.items():
        if binding.get(key) != expected:
            raise CalibrationValidationError(
                f"Calibration artifact binding mismatch for {key}."
            )
    split_path = _artifact_entry_path(split_entry, name="split manifest")
    canonical_path = _artifact_entry_path(canonical_entry, name="canonical data manifest")
    try:
        split_manifest = data_pipeline.load_split_manifest(split_path)
        canonical_data_manifest = _load_json(canonical_path, "canonical data manifest")
        data_pipeline.validate_canonical_data_manifest(canonical_data_manifest)
    except (OSError, data_pipeline.ManifestValidationError, CalibrationValidationError) as exc:
        raise CalibrationValidationError("Artifact data-manifest validation failed.") from exc
    if split_manifest.get("split_strategy") != "group" or split_manifest.get(
        "leakage_policy"
    ) != "error":
        raise CalibrationValidationError(
            "Strict inference requires a group-disjoint, leakage-free split manifest."
        )
    if binding.get("split_manifest_sha256") != split_manifest.get(
        "split_content_sha256"
    ):
        raise CalibrationValidationError("Calibration split-manifest hash mismatch.")
    canonical_hash = canonical_data_manifest.get("canonical_data_manifest_sha256")
    if (
        binding.get("canonical_data_manifest_sha256") != canonical_hash
        or split_manifest.get("canonical_data_manifest_sha256") != canonical_hash
    ):
        raise CalibrationValidationError("Calibration canonical-data-manifest hash mismatch.")
    calibration_data = calibration["calibration_data"]
    if binding.get("calibration_partition_id") != "calibration" or calibration_data.get(
        "partition_id"
    ) != "calibration":
        raise CalibrationValidationError("Strict calibration is not bound to the calibration partition.")
    if binding.get("calibration_records_sha256") != calibration_data.get(
        "records_sha256"
    ):
        raise CalibrationValidationError("Calibration record fingerprint mismatch.")
    assignment_hash = data_pipeline.split_partition_assignment_sha256(
        split_manifest, "calibration"
    )
    if (
        binding.get("calibration_partition_assignment_sha256") != assignment_hash
        or calibration_data.get("split_partition_assignment_sha256") != assignment_hash
    ):
        raise CalibrationValidationError("Calibration partition assignment fingerprint mismatch.")
    return dict(binding)


def resolve_selection_config(
    args: argparse.Namespace,
    bias_label2id: dict[str, int],
    opinion_label2id: dict[str, int],
    opinion_head_type: str | None = None,
    *,
    heads_path: Path | None = None,
    checkpoint_head_type_declared: str | None = None,
    bias_head_type: str | None = None,
    classifier_head_type: str | None = None,
    checkpoint_schema_version: str | None = None,
    opinion_style_target_mapping: dict[str, str] | None = None,
    classifier_input_contract: str | None = None,
    allowed_promotion_states: frozenset[str] = frozenset({"eligible"}),
) -> SelectionConfig:
    requested_mode = getattr(args, "selection_mode", "calibrated")
    if requested_mode not in {"auto", "calibrated", "legacy", "argmax"}:
        raise SystemExit(f"Unsupported selection mode: {requested_mode!r}.")
    model_dir = Path(args.sbert_model_dir)
    calibration_file = getattr(args, "calibration_file", None)
    calibration_path = (
        Path(calibration_file) if calibration_file else model_dir / "calibration.json"
    )
    artifact_file = getattr(args, "artifact_manifest", None)
    artifact_manifest_path = (
        Path(artifact_file) if artifact_file else model_dir / "artifact_manifest.json"
    )
    resolved_heads_path = heads_path or (
        Path(getattr(args, "classification_heads", None))
        if getattr(args, "classification_heads", None) is not None
        else model_dir / "classification_heads.pt"
    )
    bias_threshold_override = getattr(args, "bias_threshold", None)
    opinion_threshold_override = getattr(args, "opinion_threshold", None)

    if requested_mode == "argmax":
        if bias_threshold_override is not None or opinion_threshold_override is not None:
            raise SystemExit("Threshold overrides require --selection-mode legacy.")
        return SelectionConfig(
            requested_mode=requested_mode,
            effective_mode="argmax",
            calibration_file=calibration_path,
            artifact_manifest_file=artifact_manifest_path,
            calibration_reason="legacy_argmax_requested",
            legacy_compatibility=True,
        )

    if not calibration_path.exists():
        raise SystemExit(
            f"Calibration file not found: {calibration_path}. "
            "Run calibrate_sbert_heads.py or explicitly use --selection-mode argmax."
        )

    try:
        data = _load_json(calibration_path, "calibration file")
    except CalibrationValidationError as exc:
        raise SystemExit(str(exc)) from exc
    strict = requested_mode in {"auto", "calibrated"}
    if not strict and (
        data.get("schema_version") not in LEGACY_CALIBRATION_SCHEMA_VERSIONS
        and data.get("schema_version") != CALIBRATION_SCHEMA_VERSION
    ):
        raise SystemExit("Unsupported legacy calibration schema version.")
    if strict and (bias_threshold_override is not None or opinion_threshold_override is not None):
        raise SystemExit("Threshold overrides require --selection-mode legacy.")
    expected_head_type = opinion_head_type or training.LEGACY_FLAT_HEAD_TYPE
    try:
        if strict:
            if (
                bias_head_type is None
                or opinion_head_type is None
                or classifier_head_type is None
                or checkpoint_schema_version is None
                or opinion_style_target_mapping is None
                or classifier_input_contract is None
            ):
                raise CalibrationValidationError(
                    "Strict inference requires complete production checkpoint metadata."
                )
            binding = _validate_strict_artifact_binding(
                calibration=data,
                calibration_path=calibration_path,
                artifact_manifest_path=artifact_manifest_path,
                model_dir=model_dir,
                heads_path=Path(resolved_heads_path),
                bias_label2id=bias_label2id,
                opinion_label2id=opinion_label2id,
                bias_head_type=bias_head_type,
                opinion_style_head_type=opinion_head_type,
                classifier_head_type=classifier_head_type,
                checkpoint_schema_version=checkpoint_schema_version,
                opinion_style_target_mapping=opinion_style_target_mapping,
                classifier_input_contract=classifier_input_contract,
                allowed_promotion_states=allowed_promotion_states,
            )
            thresholds = data["thresholds"]
            bias_decision = _strict_decision_head_from_json(
                "bias", data["calibrators"]["bias"],
                "bias_no_clear_max", "bias_clear_min", thresholds,
            )
            opinion_style_decision = _strict_decision_head_from_json(
                "opinion_style", data["calibrators"]["opinion_style"],
                "opinion_objective_max", "opinion_opinionated_min", thresholds,
            )
            heads = None
        else:
            label_mappings = data.get("label_mappings", {})
            if not isinstance(label_mappings, dict):
                raise CalibrationValidationError("Legacy calibration label_mappings must be an object.")
            calibrated_bias_labels = label_mappings.get("bias_label2id")
            calibrated_opinion_labels = label_mappings.get("opinion_label2id")
            if (
                calibrated_bias_labels is not None
                and _label_mapping(calibrated_bias_labels, "Calibration bias_label2id")
                != bias_label2id
            ):
                raise CalibrationValidationError(
                    "Calibration bias label mapping does not match the loaded checkpoint."
                )
            if (
                calibrated_opinion_labels is not None
                and _label_mapping(calibrated_opinion_labels, "Calibration opinion_label2id")
                != opinion_label2id
            ):
                raise CalibrationValidationError(
                    "Calibration opinion label mapping does not match the loaded checkpoint."
                )
            calibrated_head_type = data.get("head_type", training.LEGACY_FLAT_HEAD_TYPE)
            if calibrated_head_type != expected_head_type:
                raise CalibrationValidationError(
                    "Calibration head type does not match the loaded checkpoint: "
                    f"calibration={calibrated_head_type!r}, checkpoint={expected_head_type!r}."
                )
            binding = None
            heads = data.get("heads") or data.get("selection") or {}
        if strict:
            bias_config = opinion_config = report_config = None
        else:
            bias_decision = opinion_style_decision = None
            bias_config = calibration_head_from_json(
                "bias", heads.get("bias"), bias_threshold_override, strict=False
            )
            opinion_config = calibration_head_from_json(
                "opinion", heads.get("opinion"), opinion_threshold_override, strict=False
            )
            report_config = calibration_head_from_json(
                "report_trigger", heads.get("report_trigger"), None, strict=False
            )
    except CalibrationValidationError as exc:
        raise SystemExit(str(exc)) from exc
    return SelectionConfig(
        requested_mode=requested_mode,
        effective_mode="calibrated" if strict else "legacy_calibrated",
        calibration_file=calibration_path,
        artifact_manifest_file=artifact_manifest_path,
        calibration_reason=(
            "manifest_bound_calibration_loaded"
            if strict
            else "legacy_calibration_requested"
        ),
        artifact_binding=binding,
        legacy_compatibility=not strict,
        bias=bias_config,
        opinion=opinion_config,
        report_trigger=report_config,
        bias_decision=bias_decision,
        opinion_style_decision=opinion_style_decision,
        include_possible_bias=bool(getattr(args, "include_possible_bias", False)),
    )


def probabilities_to_label_dict(row: Any, id2label: dict[int, str]) -> dict[str, float]:
    return {id2label[idx]: float(value) for idx, value in enumerate(row.tolist())}


def _decision_state(
    probability: float,
    config: DecisionHeadConfig,
    states: tuple[str, str, str],
) -> str:
    if config.lower.enabled and probability <= float(config.lower.threshold):
        return states[0]
    if config.upper.enabled and probability >= float(config.upper.threshold):
        return states[2]
    return states[1]


def evaluate_calibrated_decisions(
    p_bias: float,
    p_opinionated_style: float,
    selection_config: SelectionConfig,
) -> SelectionOutcome:
    if selection_config.bias_decision is None or selection_config.opinion_style_decision is None:
        raise CalibrationValidationError("Calibrated decisions require both production heads.")
    bias_assessment = _decision_state(
        p_bias,
        selection_config.bias_decision,
        ("no_clear_bias", "possible_bias", "clear_bias"),
    )
    opinion_style = _decision_state(
        p_opinionated_style,
        selection_config.opinion_style_decision,
        ("objective_style", "uncertain", "opinionated_style"),
    )
    selected = bias_assessment == "clear_bias" or (
        selection_config.include_possible_bias and bias_assessment == "possible_bias"
    )
    if bias_assessment == "clear_bias":
        selection_reasons = ["clear_bias"]
    elif selected:
        selection_reasons = ["possible_bias_explicitly_included"]
    else:
        selection_reasons = []
    abstention_reasons = [] if selected else [bias_assessment]
    return SelectionOutcome(
        selected=selected,
        selection_reasons=selection_reasons,
        abstention_reasons=abstention_reasons,
        bias_selection_score=p_bias,
        opinion_selection_score=p_opinionated_style,
        report_trigger_score=None,
        bias_positive=bias_assessment == "clear_bias",
        report_trigger=selected,
        uncertainty_status=(
            "confident" if bias_assessment in {"no_clear_bias", "clear_bias"} else "uncertain"
        ),
        bias_assessment=bias_assessment,
        opinion_style=opinion_style,
    )


def evaluate_sentence_selection(
    bias_label: str,
    opinion_label: str,
    bias_probabilities: dict[str, float],
    opinion_probabilities: dict[str, float],
    selection_config: SelectionConfig,
) -> SelectionOutcome:
    if selection_config.effective_mode == "calibrated":
        bias_score = bias_probabilities.get("Biased")
        style_score = opinion_probabilities.get("opinionated_style")
        if bias_score is None or style_score is None:
            raise CalibrationValidationError(
                "Production decisions require p_bias and p_opinionated_style."
            )
        return evaluate_calibrated_decisions(
            float(bias_score), float(style_score), selection_config
        )
    if selection_config.effective_mode == "argmax":
        reasons: list[str] = []
        if bias_label == "Biased":
            reasons.append("bias_argmax")
        if opinion_label != "Entirely factual":
            reasons.append("opinion_argmax")
        selected = bool(reasons)
        return SelectionOutcome(
            selected=selected,
            selection_reasons=reasons,
            abstention_reasons=[] if reasons else ["no_positive_label"],
            bias_selection_score=None,
            opinion_selection_score=None,
            report_trigger_score=None,
            bias_positive=bias_label == "Biased",
            report_trigger=selected,
            uncertainty_status="uncertain",
        )

    selection_reasons: list[str] = []
    gate_failures: list[str] = []
    positive_candidate = False

    bias_score = bias_probabilities.get("Biased")
    bias_positive = False
    if bias_label == "Biased":
        positive_candidate = True
        bias_config = selection_config.bias
        if bias_config is None or not bias_config.enabled:
            gate_failures.append("bias_head_disabled")
        elif bias_config.threshold is None:
            gate_failures.append("bias_missing_threshold")
        elif bias_score is not None and bias_score >= bias_config.threshold:
            bias_positive = True
        else:
            gate_failures.append("bias_below_threshold")

    factual_probability = opinion_probabilities.get("Entirely factual")
    opinion_score = None if factual_probability is None else 1.0 - factual_probability
    opinion_positive = False
    if opinion_label != "Entirely factual":
        positive_candidate = True
        opinion_config = selection_config.opinion
        if opinion_config is None or not opinion_config.enabled:
            gate_failures.append("opinion_head_disabled")
        elif opinion_config.threshold is None:
            gate_failures.append("opinion_missing_threshold")
        elif opinion_score is not None and opinion_score >= opinion_config.threshold:
            opinion_positive = True
        else:
            gate_failures.append("opinion_below_threshold")

    if not positive_candidate:
        gate_failures.append("no_positive_label")

    candidate_scores = [
        score for score in (bias_score, opinion_score) if score is not None
    ]
    report_trigger_score = max(candidate_scores) if candidate_scores else None
    report_config = selection_config.report_trigger
    if report_config is not None and report_config.enabled:
        if report_config.threshold is None:
            gate_failures.append("report_trigger_missing_threshold")
        elif (
            positive_candidate
            and report_trigger_score is not None
            and report_trigger_score >= report_config.threshold
        ):
            selection_reasons.append("report_trigger_threshold_met")
        elif positive_candidate:
            gate_failures.append("report_trigger_below_threshold")
    else:
        if bias_positive:
            selection_reasons.append("bias_threshold_met")
        if opinion_positive:
            selection_reasons.append("opinion_threshold_met")

    selected = bool(selection_reasons)
    if selection_config.legacy_compatibility:
        uncertainty_status = "uncertain"
    else:
        uncertainty_status = "confident" if selected else "abstained"

    return SelectionOutcome(
        selected=selected,
        selection_reasons=selection_reasons,
        abstention_reasons=[] if selection_reasons else gate_failures,
        bias_selection_score=bias_score,
        opinion_selection_score=opinion_score,
        report_trigger_score=report_trigger_score,
        bias_positive=bias_positive,
        report_trigger=selected,
        uncertainty_status=uncertainty_status,
    )


def selection_policy_to_json(selection_config: SelectionConfig) -> dict[str, Any]:
    record: dict[str, Any] = {
        "requested_mode": selection_config.requested_mode,
        "effective_mode": selection_config.effective_mode,
        "reason": selection_config.calibration_reason,
    }
    if selection_config.calibration_file is not None:
        record["calibration_file"] = str(selection_config.calibration_file)
    if selection_config.artifact_manifest_file is not None:
        record["artifact_manifest_file"] = str(selection_config.artifact_manifest_file)
    record["legacy_compatibility"] = selection_config.legacy_compatibility
    record["default_uncertainty_status"] = (
        "uncertain" if selection_config.legacy_compatibility else "calibrated"
    )
    if selection_config.effective_mode == "calibrated":
        record["decision_policy_version"] = DECISION_POLICY_VERSION
        record["include_possible_bias"] = selection_config.include_possible_bias
        record["bias"] = decision_head_to_json(selection_config.bias_decision)
        record["opinion_style"] = decision_head_to_json(
            selection_config.opinion_style_decision
        )
    elif selection_config.effective_mode == "legacy_calibrated":
        record["bias"] = calibration_head_to_json(selection_config.bias)
        record["opinion"] = calibration_head_to_json(selection_config.opinion)
        record["report_trigger"] = calibration_head_to_json(
            selection_config.report_trigger
        )
    if selection_config.artifact_binding is not None:
        record["artifact_binding"] = selection_config.artifact_binding
    return record


def decision_head_to_json(config: DecisionHeadConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "calibrator": {
            "method": config.calibrator.method,
            "input_kind": config.calibrator.input_kind,
            "parameters": {"temperature": config.calibrator.temperature},
        },
        "lower": threshold_region_to_json(config.lower),
        "upper": threshold_region_to_json(config.upper),
    }


def threshold_region_to_json(config: ThresholdRegionConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "threshold": config.threshold,
        "metric": config.metric,
        "target": config.target,
        "confidence": config.confidence,
        "minimum_predicted_support": config.minimum_predicted_support,
        "selected_support": config.selected_support,
        "observed_metric": config.observed_metric,
        "wilson_lower_bound": config.wilson_lower_bound,
        "coverage": config.coverage,
        "disabled_reason": config.disabled_reason,
    }


def calibration_head_to_json(config: CalibrationHeadConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "enabled": config.enabled,
        "threshold": config.threshold,
        "temperature": config.temperature,
        "target_precision": config.target_precision,
        "validation_precision": config.validation_precision,
        "validation_recall": config.validation_recall,
        "validation_selected_count": config.validation_selected_count,
        "validation_coverage": config.validation_coverage,
        "validation_candidate_count": config.validation_candidate_count,
        "validation_candidate_coverage": config.validation_candidate_coverage,
        "validation_positive_support": config.validation_positive_support,
        "validation_precision_wilson_lower_bound": (
            config.validation_precision_wilson_lower_bound
        ),
        "precision_confidence": config.precision_confidence,
        "coverage_unit": config.coverage_unit,
        "disabled_reason": config.disabled_reason,
    }


def build_abstention_summary(predictions: list[SentencePrediction]) -> dict[str, Any]:
    abstention_reasons = Counter(
        reason
        for prediction in predictions
        if not prediction.selected
        for reason in (prediction.abstention_reasons or [])
    )
    selection_reasons = Counter(
        reason
        for prediction in predictions
        if prediction.selected
        for reason in (prediction.selection_reasons or [])
    )
    uncertainty_statuses = Counter(
        prediction.uncertainty_status for prediction in predictions
    )
    bias_assessments = Counter(
        prediction.bias_assessment
        for prediction in predictions
        if prediction.bias_assessment is not None
    )
    opinion_styles = Counter(
        prediction.opinion_style
        for prediction in predictions
        if prediction.opinion_style is not None
    )
    selected_count = sum(1 for prediction in predictions if prediction.selected)
    candidate_abstained_count = sum(
        1
        for prediction in predictions
        if not prediction.selected
        and any(reason != "no_positive_label" for reason in (prediction.abstention_reasons or []))
    )
    return {
        "sentence_count": len(predictions),
        "selected_sentence_count": selected_count,
        "abstained_sentence_count": len(predictions) - selected_count,
        "candidate_abstained_count": candidate_abstained_count,
        "selection_reasons": dict(sorted(selection_reasons.items())),
        "abstention_reasons": dict(sorted(abstention_reasons.items())),
        "uncertainty_statuses": dict(sorted(uncertainty_statuses.items())),
        "bias_assessments": dict(sorted(bias_assessments.items())),
        "opinion_styles": dict(sorted(opinion_styles.items())),
    }


def classify_sentences(
    sentences: list[str] | list[SentenceContextInput],
    model: Any,
    bias_head: Any,
    opinion_head: Any,
    bias_label2id: dict[str, int],
    opinion_label2id: dict[str, int],
    device: Any,
    batch_size: int,
    selection_config: SelectionConfig | None = None,
) -> list[SentencePrediction]:
    torch = training.torch
    assert torch is not None

    if selection_config is None:
        raise ValueError(
            "selection_config is required; resolve an explicit calibrated or legacy mode."
        )
    id2bias = {idx: label for label, idx in bias_label2id.items()}
    id2opinion = {idx: label for label, idx in opinion_label2id.items()}
    predictions: list[SentencePrediction] = []
    sentence_inputs = _normalize_sentence_inputs(sentences)

    with torch.no_grad():
        for batch_start in range(0, len(sentence_inputs), batch_size):
            batch_inputs = sentence_inputs[batch_start : batch_start + batch_size]
            embeddings = training.sentence_embeddings(
                model, [item.model_input for item in batch_inputs], device
            )
            bias_logits = bias_head(embeddings)
            opinion_outputs = opinion_head(embeddings)
            opinion_head_type = getattr(
                opinion_head, "head_type", training.LEGACY_FLAT_HEAD_TYPE
            )
            if selection_config.effective_mode == "calibrated":
                if (
                    getattr(bias_head, "head_type", None) != training.BINARY_BIAS_HEAD_TYPE
                    or opinion_head_type != training.BINARY_OPINION_STYLE_HEAD_TYPE
                    or bias_logits.shape[-1] != 1
                    or opinion_outputs.shape[-1] != 1
                ):
                    raise CalibrationValidationError(
                        "Production calibrated inference requires two scalar binary logits."
                    )
                assert selection_config.bias_decision is not None
                assert selection_config.opinion_style_decision is not None
                bias_raw_logits = bias_logits.reshape(-1)
                style_raw_logits = opinion_outputs.reshape(-1)
                raw_bias_probabilities = torch.sigmoid(bias_raw_logits).detach().cpu()
                raw_style_probabilities = torch.sigmoid(style_raw_logits).detach().cpu()
                p_bias_values = torch.sigmoid(
                    bias_raw_logits
                    / selection_config.bias_decision.calibrator.temperature
                ).detach().cpu()
                p_style_values = torch.sigmoid(
                    style_raw_logits
                    / selection_config.opinion_style_decision.calibrator.temperature
                ).detach().cpu()
                for offset, sentence_input in enumerate(batch_inputs):
                    sentence_index = sentence_input.index
                    sentence = sentence_input.text
                    raw_p_bias = float(raw_bias_probabilities[offset].item())
                    raw_p_style = float(raw_style_probabilities[offset].item())
                    p_bias = float(p_bias_values[offset].item())
                    p_style = float(p_style_values[offset].item())
                    bias_label = "Biased" if raw_p_bias >= 0.5 else "Non-biased"
                    opinion_label = (
                        "opinionated_style" if raw_p_style >= 0.5 else "objective_style"
                    )
                    selection = evaluate_calibrated_decisions(
                        p_bias, p_style, selection_config
                    )
                    predictions.append(
                        SentencePrediction(
                            index=sentence_index,
                            text=sentence,
                            bias_label=bias_label,
                            bias_probability=max(raw_p_bias, 1.0 - raw_p_bias),
                            bias_probabilities={
                                "Non-biased": 1.0 - raw_p_bias,
                                "Biased": raw_p_bias,
                            },
                            opinion_label=opinion_label,
                            opinion_probability=max(raw_p_style, 1.0 - raw_p_style),
                            opinion_probabilities={
                                "objective_style": 1.0 - raw_p_style,
                                "opinionated_style": raw_p_style,
                            },
                            selected=selection.selected,
                            selection_reasons=selection.selection_reasons,
                            abstention_reasons=selection.abstention_reasons,
                            bias_selection_score=p_bias,
                            opinion_selection_score=p_style,
                            bias_calibrated_probability=p_bias,
                            bias_calibrated_probabilities={
                                "Non-biased": 1.0 - p_bias,
                                "Biased": p_bias,
                            },
                            opinion_calibrated_probability=p_style,
                            opinion_calibrated_probabilities={
                                "objective_style": 1.0 - p_style,
                                "opinionated_style": p_style,
                            },
                            bias_positive=selection.bias_positive,
                            report_trigger=selection.report_trigger,
                            uncertainty_status=selection.uncertainty_status,
                            bias_assessment=selection.bias_assessment,
                            opinion_style=selection.opinion_style,
                            p_bias=p_bias,
                            p_opinionated_style=p_style,
                            bias_logit=float(bias_raw_logits[offset].detach().cpu().item()),
                            opinionated_style_logit=float(
                                style_raw_logits[offset].detach().cpu().item()
                            ),
                            legacy_compatibility=False,
                            context_text=sentence_input.context_text,
                            model_input=sentence_input.model_input,
                        )
                    )
                continue
            bias_probs = torch.softmax(bias_logits, dim=-1).detach().cpu()
            opinion_probs = training.opinion_probabilities_from_outputs(
                opinion_outputs, head_type=opinion_head_type
            ).detach().cpu()

            bias_calibrated_probs = None
            opinion_calibrated_probs = None
            if selection_config.effective_mode in {"calibrated", "legacy_calibrated"}:
                bias_temperature = (
                    selection_config.bias.temperature
                    if selection_config.bias is not None
                    else 1.0
                )
                opinion_temperature = (
                    selection_config.opinion.temperature
                    if selection_config.opinion is not None
                    else 1.0
                )
                bias_calibrated_probs = torch.softmax(
                    bias_logits / bias_temperature, dim=-1
                ).detach().cpu()
                opinion_calibrated_probs = training.opinion_probabilities_from_outputs(
                    opinion_outputs,
                    head_type=opinion_head_type,
                    temperature=opinion_temperature,
                ).detach().cpu()

            for offset, sentence_input in enumerate(batch_inputs):
                sentence_index = sentence_input.index
                sentence = sentence_input.text
                bias_row = bias_probs[offset]
                opinion_row = opinion_probs[offset]
                bias_id = int(bias_row.argmax().item())
                opinion_id = int(opinion_row.argmax().item())
                bias_label = id2bias[bias_id]
                opinion_label = id2opinion[opinion_id]
                bias_probability = float(bias_row[bias_id].item())
                opinion_probability = float(opinion_row[opinion_id].item())
                bias_probabilities = probabilities_to_label_dict(bias_row, id2bias)
                opinion_probabilities = probabilities_to_label_dict(opinion_row, id2opinion)

                bias_calibrated_probability = None
                bias_calibrated_probability_map = None
                opinion_calibrated_probability = None
                opinion_calibrated_probability_map = None
                bias_calibrated_label = None
                opinion_calibrated_label = None
                selection_bias_label = bias_label
                selection_opinion_label = opinion_label
                selection_bias_probabilities = bias_probabilities
                selection_opinion_probabilities = opinion_probabilities
                if bias_calibrated_probs is not None and opinion_calibrated_probs is not None:
                    bias_calibrated_row = bias_calibrated_probs[offset]
                    opinion_calibrated_row = opinion_calibrated_probs[offset]
                    bias_calibrated_id = int(bias_calibrated_row.argmax().item())
                    opinion_calibrated_id = int(opinion_calibrated_row.argmax().item())
                    bias_calibrated_label = id2bias[bias_calibrated_id]
                    opinion_calibrated_label = id2opinion[opinion_calibrated_id]
                    bias_calibrated_probability = float(
                        bias_calibrated_row[bias_calibrated_id].item()
                    )
                    opinion_calibrated_probability = float(
                        opinion_calibrated_row[opinion_calibrated_id].item()
                    )
                    bias_calibrated_probability_map = probabilities_to_label_dict(
                        bias_calibrated_row, id2bias
                    )
                    opinion_calibrated_probability_map = probabilities_to_label_dict(
                        opinion_calibrated_row, id2opinion
                    )
                    selection_bias_probabilities = bias_calibrated_probability_map
                    selection_opinion_probabilities = opinion_calibrated_probability_map
                    selection_bias_label = bias_calibrated_label
                    selection_opinion_label = opinion_calibrated_label

                selection = evaluate_sentence_selection(
                    bias_label=selection_bias_label,
                    opinion_label=selection_opinion_label,
                    bias_probabilities=selection_bias_probabilities,
                    opinion_probabilities=selection_opinion_probabilities,
                    selection_config=selection_config,
                )

                predictions.append(
                    SentencePrediction(
                        index=sentence_index,
                        text=sentence,
                        bias_label=bias_label,
                        bias_probability=bias_probability,
                        bias_probabilities=bias_probabilities,
                        opinion_label=opinion_label,
                        opinion_probability=opinion_probability,
                        opinion_probabilities=opinion_probabilities,
                        selected=selection.selected,
                        selection_reasons=selection.selection_reasons,
                        abstention_reasons=selection.abstention_reasons,
                        bias_selection_score=selection.bias_selection_score,
                        opinion_selection_score=selection.opinion_selection_score,
                        bias_calibrated_probability=bias_calibrated_probability,
                        bias_calibrated_probabilities=bias_calibrated_probability_map,
                        opinion_calibrated_probability=opinion_calibrated_probability,
                        opinion_calibrated_probabilities=opinion_calibrated_probability_map,
                        bias_calibrated_label=bias_calibrated_label,
                        opinion_calibrated_label=opinion_calibrated_label,
                        bias_positive=selection.bias_positive,
                        report_trigger=selection.report_trigger,
                        report_trigger_score=selection.report_trigger_score,
                        uncertainty_status=selection.uncertainty_status,
                        context_text=sentence_input.context_text,
                        model_input=sentence_input.model_input,
                    )
                )

    return predictions


def build_context_windows(
    sentences: list[str],
    selected_indices: list[int],
    context_sentences: int,
    max_windows: int,
) -> list[ContextWindow]:
    if not selected_indices:
        return []

    sentence_count = len(sentences)
    raw_windows = [
        (
            max(0, index - context_sentences),
            min(sentence_count - 1, index + context_sentences),
            [index],
        )
        for index in sorted(set(selected_indices))
    ]

    merged: list[tuple[int, int, list[int]]] = []
    for start, end, indices in raw_windows:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end, indices))
            continue
        previous_start, previous_end, previous_indices = merged[-1]
        merged[-1] = (
            previous_start,
            max(previous_end, end),
            sorted(set(previous_indices + indices)),
        )

    if max_windows > 0:
        merged = merged[:max_windows]

    return [
        ContextWindow(
            start_index=start,
            end_index=end,
            selected_sentence_indices=indices,
            text=" ".join(sentences[start : end + 1]),
        )
        for start, end, indices in merged
    ]


def evidence_items_for_prompt(
    evidence_items: list[evidence_retrieval.EvidenceItem],
) -> list[dict[str, Any]]:
    prompt_items: list[dict[str, Any]] = []
    for index, item in enumerate(evidence_items, start=1):
        prompt_items.append(
            {
                "citation_id": f"E{index}",
                "title": item.title,
                "url": item.url,
                "domain": item.domain,
                "source_type": item.source_type,
                "published_date": item.published_date,
                "snippet": item.snippet,
            }
        )
    return prompt_items


def build_prompt(
    window: ContextWindow,
    predictions_by_index: dict[int, SentencePrediction],
    evidence_items: list[evidence_retrieval.EvidenceItem] | None = None,
    *,
    citations: list[evidence_assessment.CitationRecord] | None = None,
) -> str:
    if citations is None and evidence_items:
        registry = evidence_assessment.CitationRegistry()
        citations = registry.register(evidence_items)
    reviewed_sentences = [
        {
            "index": index,
            "text": predictions_by_index[index].text,
            "context": predictions_by_index[index].context_text or window.text,
            "bias_assessment": predictions_by_index[index].bias_assessment,
            "opinion_style": predictions_by_index[index].opinion_style,
            "evidence_status": predictions_by_index[index].evidence_status,
            "claims": predictions_by_index[index].claims or [],
        }
        for index in window.selected_sentence_indices
    ]
    reviewed_json = json.dumps(reviewed_sentences, indent=2)
    base_prompt = (
        "Analyze only the article passage below. The sentences listed under "
        "sentences_for_review are the article sentences that need focused analysis. "
        "Use the surrounding_article_context only to interpret those sentences in "
        "their original chronological context.\n\n"
        f"sentences_for_review:\n{reviewed_json}\n\n"
        f"surrounding_article_context:\n{window.text}"
    )

    citation_records = [citation.to_json() for citation in (citations or [])]
    if not citation_records:
        return (
            f"{base_prompt}\n\n"
            "No external citations are available. Do not invent citations, URLs, "
            "support, contradiction, or factual verification. Discuss only wording, "
            "framing, and the supplied evidence_status values."
        )

    evidence_json = json.dumps(citation_records, indent=2)
    return (
        f"{base_prompt}\n\n"
        "trusted_external_evidence:\n"
        f"{evidence_json}\n\n"
        "Use only the completed claim evidence_status and its citation_ids when "
        "describing support or contradiction. Retrieval scores are ranking metadata, "
        "not evidence findings. search_snippet material is discovery-only. Cite only "
        "the supplied citation_id values and never invent a citation or URL."
    )


class LlamaGenerator:
    def __init__(
        self,
        model_name: str,
        hf_token: str | None,
        allow_cpu: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        revision: str | None = DEFAULT_LLM_REVISION,
    ) -> None:
        torch = training.torch
        assert torch is not None

        if not torch.cuda.is_available() and not allow_cpu:
            raise SystemExit(
                "CUDA is not available to this Python environment. "
                "Llama 3.1 8B local inference requires a CUDA-enabled Torch install, "
                "or rerun with --allow-cpu-llm for a slow CPU fallback. "
                "Use --prompt-only to validate SBERT classification and prompt output."
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.revision = revision

        token = hf_token or os.environ.get("HF_TOKEN")
        LOGGER.info("Loading LLM %s at revision %s", model_name, revision or "default")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            revision=revision,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"token": token, "revision": revision}
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()

    def generate(self, prompt: str) -> str:
        return self.generate_with_system(MEDIA_BIAS_SYSTEM_PROMPT, prompt)

    def generate_with_system(self, system_prompt: str, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            template_output = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_device = next(self.model.parameters()).device
            if isinstance(template_output, Mapping):
                input_length = template_output["input_ids"].shape[-1]
                generate_kwargs = {
                    key: value.to(input_device) if hasattr(value, "to") else value
                    for key, value in template_output.items()
                }
            else:
                input_length = template_output.shape[-1]
                input_ids = template_output.to(input_device)
                generate_kwargs = {"input_ids": input_ids}
        else:
            rendered = (
                f"System: {messages[0]['content']}\n\n"
                f"User: {messages[1]['content']}\n\nAssistant:"
            )
            inputs = self.tokenizer(rendered, return_tensors="pt")
            input_length = inputs["input_ids"].shape[-1]
            input_device = next(self.model.parameters()).device
            generate_kwargs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

        do_sample = self.temperature > 0
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **generate_kwargs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][input_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def normalize_llm_report(text: str) -> str:
    return text.strip()


def prediction_to_json(
    prediction: SentencePrediction, *, include_evidence: bool = False
) -> dict[str, Any]:
    if not prediction.legacy_compatibility:
        record: dict[str, Any] = {
            "index": prediction.index,
            "text": prediction.text,
            "selected": prediction.selected,
            "bias_assessment": prediction.bias_assessment,
            "opinion_style": prediction.opinion_style,
            "p_bias": prediction.p_bias,
            "p_opinionated_style": prediction.p_opinionated_style,
            "diagnostics": {
                "bias_logit": prediction.bias_logit,
                "opinionated_style_logit": prediction.opinionated_style_logit,
            },
        }
        if include_evidence:
            record["evidence_status"] = prediction.evidence_status
            record["claims"] = prediction.claims or []
        if prediction.selection_reasons is not None:
            record["selection_reasons"] = prediction.selection_reasons
        if prediction.abstention_reasons:
            record["abstention_reasons"] = prediction.abstention_reasons
        return record
    record: dict[str, Any] = {
        "index": prediction.index,
        "text": prediction.text,
        "selected": prediction.selected,
        "bias_label": prediction.bias_label,
        "bias_probability": prediction.bias_probability,
        "bias_probabilities": prediction.bias_probabilities,
        "opinion_label": prediction.opinion_label,
        "opinion_probability": prediction.opinion_probability,
        "opinion_probabilities": prediction.opinion_probabilities,
        "uncertainty_status": prediction.uncertainty_status,
    }
    if prediction.selection_reasons is not None:
        record["selection_reasons"] = prediction.selection_reasons
    if prediction.abstention_reasons:
        record["abstention_reasons"] = prediction.abstention_reasons
    if prediction.bias_selection_score is not None:
        record["bias_selection_score"] = prediction.bias_selection_score
    if prediction.opinion_selection_score is not None:
        record["opinion_selection_score"] = prediction.opinion_selection_score
    if prediction.bias_calibrated_probability is not None:
        record["bias_calibrated_probability"] = prediction.bias_calibrated_probability
    if prediction.bias_calibrated_probabilities is not None:
        record["bias_calibrated_probabilities"] = prediction.bias_calibrated_probabilities
    if prediction.opinion_calibrated_probability is not None:
        record["opinion_calibrated_probability"] = prediction.opinion_calibrated_probability
    if prediction.opinion_calibrated_probabilities is not None:
        record["opinion_calibrated_probabilities"] = prediction.opinion_calibrated_probabilities
    if prediction.bias_calibrated_label is not None:
        record["bias_calibrated_label"] = prediction.bias_calibrated_label
    if prediction.opinion_calibrated_label is not None:
        record["opinion_calibrated_label"] = prediction.opinion_calibrated_label
    if prediction.bias_positive is not None:
        record["bias_positive"] = prediction.bias_positive
    if prediction.report_trigger is not None:
        record["report_trigger"] = prediction.report_trigger
    if prediction.report_trigger_score is not None:
        record["report_trigger_score"] = prediction.report_trigger_score
    return record


def analyze_article(
    article: ArticleRecord,
    model: Any,
    bias_head: Any,
    opinion_head: Any,
    bias_label2id: dict[str, int],
    opinion_label2id: dict[str, int],
    device: Any,
    args: argparse.Namespace,
    llm: LlamaGenerator | None,
    evidence_policy: evidence_retrieval.TrustedSourcePolicy | None = None,
    selection_config: SelectionConfig | None = None,
) -> dict[str, Any]:
    if selection_config is None:
        raise ValueError(
            "selection_config is required; resolve an explicit calibrated or legacy mode."
        )
    sentences = split_sentences(article.text)
    sentence_inputs = build_sentence_context_inputs(
        sentences, int(getattr(args, "context_sentences", 2))
    )
    predictions = classify_sentences(
        sentences=sentence_inputs,
        model=model,
        bias_head=bias_head,
        opinion_head=opinion_head,
        bias_label2id=bias_label2id,
        opinion_label2id=opinion_label2id,
        device=device,
        batch_size=args.batch_size,
        selection_config=selection_config,
    )
    selected_predictions = [prediction for prediction in predictions if prediction.selected]
    selected_indices = [prediction.index for prediction in selected_predictions]
    windows = build_context_windows(
        sentences=sentences,
        selected_indices=selected_indices,
        context_sentences=args.context_sentences,
        max_windows=args.max_windows_per_article,
    )
    predictions_by_index = {prediction.index: prediction for prediction in predictions}
    evidence_mode = resolve_evidence_mode(args)
    generate = (
        None
        if llm is None
        else lambda system_prompt, prompt: llm.generate_with_system(system_prompt, prompt)
    )
    extractions = evidence_assessment.extract_verifiable_claims(
        article_id=article.article_id,
        candidates=[
            {
                "sentence_index": prediction.index,
                "sentence": prediction.text,
                "context": prediction.context_text,
            }
            for prediction in selected_predictions
        ],
        generate=generate,
    )
    extraction_by_index = {
        extraction.sentence_index: extraction for extraction in extractions
    }
    for extraction in extractions:
        prediction = predictions_by_index[extraction.sentence_index]
        prediction.claims = []
        prediction.evidence_status = evidence_assessment.sentence_evidence_status(
            extraction, []
        )
    citation_registry = evidence_assessment.CitationRegistry()

    context_windows: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        evidence_result = evidence_retrieval.EvidenceResult(
            status="not_requested",
            items=[],
            requested_mode=evidence_mode,
            effective_mode="off",
            elapsed_ms=0,
            request_id=f"{article.article_id}:window:{window_index}",
            article_id=article.article_id,
            policy_version=(
                evidence_policy or evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY
            ).version,
        )
        window_extractions = [
            extraction_by_index[index]
            for index in window.selected_sentence_indices
            if index in extraction_by_index
        ]
        window_claims = [
            claim
            for extraction in window_extractions
            if extraction.verifiability == "verifiable"
            for claim in extraction.claims
        ]
        if evidence_mode != "off" and window_claims:
            selected_texts = [
                predictions_by_index[index].text
                for index in window.selected_sentence_indices
            ]
            evidence_claims = tuple(
                evidence_retrieval.EvidenceClaim(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    context_text=claim.context_text,
                    selected_sentence_indices=(claim.sentence_index,),
                )
                for claim in window_claims
            )
            evidence_result = evidence_retrieval.retrieve_evidence(
                selected_sentence_texts=selected_texts,
                context_text=window.text,
                provider=getattr(args, "evidence_provider", "brave"),
                max_items=getattr(args, "max_evidence_items", 5),
                timeout_seconds=getattr(args, "evidence_timeout_seconds", 10.0),
                policy=evidence_policy
                or evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY,
                mode=evidence_mode,
                request_id=f"{article.article_id}:window:{window_index}",
                article_id=article.article_id,
                claims=evidence_claims,
            )

        window_citations = citation_registry.register(evidence_result.items)
        assessments = evidence_assessment.assess_claims(
            claims=window_claims,
            retrieval_status=evidence_result.status,
            citations=window_citations,
            generate=generate,
        )
        assessments_by_sentence: dict[int, list[evidence_assessment.ClaimAssessment]] = {}
        for assessment in assessments:
            assessments_by_sentence.setdefault(assessment.sentence_index, []).append(assessment)
        for extraction in window_extractions:
            prediction = predictions_by_index[extraction.sentence_index]
            sentence_assessments = assessments_by_sentence.get(extraction.sentence_index, [])
            prediction.claims = [item.to_json() for item in sentence_assessments]
            prediction.evidence_status = evidence_assessment.sentence_evidence_status(
                extraction, sentence_assessments
            )

        prompt = build_prompt(
            window,
            predictions_by_index,
            citations=window_citations,
        )
        report = None
        if llm is not None and not bool(getattr(args, "defer_report_generation", False)):
            report = normalize_llm_report(llm.generate(prompt))

        window_record = {
            "window_index": window_index,
            "start_index": window.start_index,
            "end_index": window.end_index,
            "selected_sentence_indices": window.selected_sentence_indices,
            "text": window.text,
            "retrieval_status": evidence_result.status,
            "sentence_evidence_statuses": {
                str(index): predictions_by_index[index].evidence_status
                for index in window.selected_sentence_indices
            },
            "claims": [assessment.to_json() for assessment in assessments],
            "evidence_items": [citation.to_json() for citation in window_citations],
            "evidence_metadata": evidence_result.metadata_json(),
            "prompt": prompt,
            "llm_report": report,
        }
        if evidence_result.error:
            window_record["evidence_error"] = evidence_result.error
        context_windows.append(window_record)

    record: dict[str, Any] = {
        "article_id": article.article_id,
        "metadata": article.metadata,
        "sbert_model_dir": str(args.sbert_model_dir),
        "llm_model": args.llm_model,
        "llm_revision": getattr(args, "llm_revision", DEFAULT_LLM_REVISION),
        "sentence_count": len(sentences),
        "selected_sentence_count": len(selected_predictions),
        "selected_sentences": [
            prediction_to_json(prediction, include_evidence=True)
            for prediction in selected_predictions
        ],
        "evidence": [citation.to_json() for citation in citation_registry.records()],
        "selection_policy": selection_policy_to_json(selection_config),
        "abstention_summary": build_abstention_summary(predictions),
        "context_windows": context_windows,
    }
    if getattr(args, "include_all_sentence_predictions", False):
        record["sentence_predictions"] = [
            prediction_to_json(prediction, include_evidence=True)
            for prediction in predictions
        ]
    return record


def write_records(records: list[dict[str, Any]], output_file: Path | str) -> None:
    if str(output_file) == "-":
        for record in records:
            print(json.dumps(record))
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    args = parse_args()
    try:
        runtime_config.load_explicit_env_file(args.env_file)
        if args.evidence_mode != "off":
            runtime_config.require_runtime_readiness(args.evidence_mode)
    except runtime_config.RuntimeConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    require_inference_dependencies(prompt_only=args.prompt_only)

    torch = training.torch
    assert torch is not None
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using SBERT device: %s", device)

    articles = load_article_records(args)
    if not articles:
        raise SystemExit("No non-empty article text found in input.")

    evidence_policy = evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY
    if args.evidence_mode in {"web", "hybrid"}:
        evidence_policy = evidence_retrieval.load_trusted_source_policy(
            args.trusted_sources_file
        )
    if args.evidence_mode in {"web", "hybrid"}:
        LOGGER.info("Trusted web evidence retrieval enabled with provider: %s", args.evidence_provider)
    elif args.evidence_mode == "corpus":
        LOGGER.info("Trusted corpus evidence retrieval enabled through QDRANT_* settings")

    model, bias_head, opinion_head, bias_label2id, opinion_label2id = load_sbert_and_heads(
        model_dir=args.sbert_model_dir,
        heads_path=args.classification_heads,
        device=device,
        compatibility_mode=(
            "legacy" if args.selection_mode in {"legacy", "argmax"} else "strict"
        ),
    )
    heads_path = args.classification_heads or (
        args.sbert_model_dir / "classification_heads.pt"
    )
    selection_config = resolve_selection_config(
        args,
        bias_label2id,
        opinion_label2id,
        getattr(opinion_head, "head_type", training.LEGACY_FLAT_HEAD_TYPE),
        heads_path=heads_path,
        checkpoint_head_type_declared=getattr(
            opinion_head, "checkpoint_declared_head_type", None
        ),
        bias_head_type=getattr(bias_head, "head_type", None),
        classifier_head_type=getattr(opinion_head, "classifier_head_type", None),
        checkpoint_schema_version=getattr(
            opinion_head, "checkpoint_schema_version", None
        ),
        opinion_style_target_mapping=getattr(
            opinion_head, "opinion_style_target_mapping", None
        ),
        classifier_input_contract=getattr(
            opinion_head, "classifier_input_contract", None
        ),
    )
    LOGGER.info("Sentence selection mode: %s", selection_config.effective_mode)

    llm = None
    if not args.prompt_only:
        llm = LlamaGenerator(
            model_name=args.llm_model,
            hf_token=args.hf_token,
            allow_cpu=args.allow_cpu_llm,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            revision=args.llm_revision,
        )

    records = [
        analyze_article(
            article=article,
            model=model,
            bias_head=bias_head,
            opinion_head=opinion_head,
            bias_label2id=bias_label2id,
            opinion_label2id=opinion_label2id,
            device=device,
            args=args,
            llm=llm,
            evidence_policy=evidence_policy,
            selection_config=selection_config,
        )
        for article in articles
    ]
    write_records(records, args.output_file)

    analyzed = len(records)
    selected = sum(record["selected_sentence_count"] for record in records)
    windows = sum(len(record["context_windows"]) for record in records)
    LOGGER.info(
        "Analyzed %d article(s), selected %d sentence(s), built %d context window(s).",
        analyzed,
        selected,
        windows,
    )
    if str(args.output_file) != "-":
        LOGGER.info("Wrote results to %s", args.output_file)


if __name__ == "__main__":
    main()
