#!/usr/bin/env python
"""Fit manifest-bound binary calibration and decision thresholds."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
LLM_DIR = PROJECT_ROOT / "LLM-inference"
for import_path in (PROJECT_ROOT, LLM_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import artifact_manifest
import data_pipeline
import evaluation as foundation_evaluation
import finetune_all_mpnet_babe as training
import infer_bias_llm

LOGGER = logging.getLogger("calibrate_sbert_heads")

CALIBRATION_SCHEMA_VERSION = "calibration/v3"
CALIBRATION_DATA_SCHEMA_VERSION = "calibration_partition/v1"
CALIBRATOR_SCHEMA_VERSION = "binary_calibrator/v1"
DIAGNOSTICS_SCHEMA_VERSION = "binary_calibration_diagnostics/v1"
DECISION_POLICY_VERSION = "classifier_decision/v1"
DEFAULT_DIAGNOSTIC_BINS = 10


def parse_args() -> argparse.Namespace:
    default_model = PROJECT_ROOT / "models" / "all-mpnet-base-v2-babe" / "best"
    parser = argparse.ArgumentParser(
        description="Fit binary temperature scaling and Wilson-LCB decision regions."
    )
    parser.add_argument("--sbert-model-dir", type=Path, default=default_model)
    parser.add_argument("--classification-heads", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--bias-target-precision", type=float, default=0.90)
    parser.add_argument("--bias-target-npv", type=float, default=0.90)
    parser.add_argument("--opinion-target-precision", type=float, default=0.90)
    parser.add_argument("--opinion-target-npv", type=float, default=0.90)
    parser.add_argument("--minimum-predicted-support", type=int, default=30)
    parser.add_argument("--precision-confidence", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--data-glob", nargs="+", default=None)
    parser.add_argument("--train-data-glob", nargs="+", default=None)
    parser.add_argument("--test-data-glob", nargs="+", default=None)
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--train-split-value", default=None)
    parser.add_argument("--test-split-value", default=None)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--bias-label-column", default=None)
    parser.add_argument("--opinion-label-column", default=None)
    parser.add_argument(
        "--include-no-agreement", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--missing-label-policy", choices=["mask", "drop"], default=None)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--canonical-data-manifest", type=Path, default=None)
    parser.add_argument("--artifact-manifest", type=Path, default=None)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--article-id-column", default=None)
    parser.add_argument("--event-column", default=None)
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else PROJECT_ROOT / result


def apply_metadata_defaults(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = args.sbert_model_dir / "training_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    defaults = {
        "data_dir": resolve_project_path(metadata.get("data_dir", "BABE_HF")),
        "data_glob": metadata.get("data_glob") or ["final_labels_*.csv"],
        "train_data_glob": metadata.get("train_data_glob"),
        "test_data_glob": metadata.get("test_data_glob"),
        "split_column": metadata.get("split_column"),
        "train_split_value": metadata.get("train_split_value", "train"),
        "test_split_value": metadata.get("test_split_value", "test"),
        "text_column": metadata.get("text_column", "text"),
        "bias_label_column": metadata.get("bias_label_column", "label_bias"),
        "opinion_label_column": metadata.get("opinion_label_column", "label_opinion"),
        "include_no_agreement": bool(metadata.get("include_no_agreement", False)),
        "missing_label_policy": metadata.get("missing_label_policy", "mask"),
        "seed": int(metadata.get("seed", 42)),
        "batch_size": int(metadata.get("batch_size", 16)),
        "precision_confidence": float(metadata.get("precision_confidence", 0.95)),
        "group_column": metadata.get("group_column", "news_link"),
        "article_id_column": metadata.get("article_id_column"),
        "event_column": metadata.get("event_column"),
    }
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    if args.output_file is None:
        args.output_file = args.sbert_model_dir / "calibration.json"
    if args.split_manifest is None and metadata.get("split_manifest_path"):
        args.split_manifest = resolve_project_path(metadata["split_manifest_path"])
    if args.canonical_data_manifest is None and metadata.get(
        "canonical_data_manifest_path"
    ):
        args.canonical_data_manifest = resolve_project_path(
            metadata["canonical_data_manifest_path"]
        )
    if args.artifact_manifest is None:
        args.artifact_manifest = args.sbert_model_dir / "artifact_manifest.json"
    args.classifier_mode = training.PRODUCTION_CLASSIFIER_MODE
    args.opinion_head_type = None
    return metadata


def _validate_search_inputs(
    scores: Sequence[float], labels: Sequence[int], target: float,
    minimum_support: int, confidence: float,
) -> None:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    if not 0.0 <= target <= 1.0:
        raise ValueError("target must be between zero and one.")
    if minimum_support <= 0:
        raise ValueError("minimum_support must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one.")
    if any(not math.isfinite(float(value)) for value in scores):
        raise ValueError("threshold scores must all be finite.")
    if any(int(value) not in (0, 1) for value in labels):
        raise ValueError("threshold labels must be binary.")


def select_confident_region(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    direction: str,
    target: float,
    minimum_support: int,
    confidence: float = foundation_evaluation.DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Select one inclusive upper/lower region using a one-sided Wilson LCB."""

    _validate_search_inputs(scores, labels, target, minimum_support, confidence)
    if direction not in {"upper", "lower"}:
        raise ValueError("direction must be 'upper' or 'lower'.")
    metric = "precision" if direction == "upper" else "negative_predictive_value"
    correct_label = 1 if direction == "upper" else 0
    total = len(scores)
    thresholds = sorted({float(value) for value in scores}, reverse=direction == "lower")
    attempts: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = [
            index
            for index, score in enumerate(scores)
            if (float(score) >= threshold if direction == "upper" else float(score) <= threshold)
        ]
        support = len(selected)
        correct = sum(int(labels[index]) == correct_label for index in selected)
        observed = correct / support if support else None
        lcb = foundation_evaluation.one_sided_wilson_lower_bound(
            correct, support, confidence
        )
        attempts.append(
            {
                "threshold": threshold,
                "selected_support": support,
                "correct_support": correct,
                "observed_metric": observed,
                "wilson_lower_bound": lcb,
                "coverage": support / total if total else 0.0,
            }
        )
    qualifying = [
        item
        for item in attempts
        if item["selected_support"] >= minimum_support
        and item["wilson_lower_bound"] >= target
    ]
    selected_attempt = qualifying[0] if qualifying else None
    reference_support = sum(int(value) == correct_label for value in labels)
    if selected_attempt is not None:
        disabled_reason = None
    elif total == 0:
        disabled_reason = "no_supervised_calibration_examples"
    elif reference_support == 0:
        disabled_reason = (
            "no_positive_reference_examples"
            if direction == "upper"
            else "no_negative_reference_examples"
        )
    elif total < minimum_support:
        disabled_reason = "minimum_predicted_support_not_met"
    else:
        disabled_reason = "target_wilson_lcb_not_met"
    best_attempt = None
    if attempts:
        best_attempt = sorted(
            attempts,
            key=lambda item: (
                -float(item["wilson_lower_bound"]),
                -int(item["selected_support"]),
                float(item["threshold"])
                if direction == "upper"
                else -float(item["threshold"]),
            ),
        )[0]
    metrics = selected_attempt or best_attempt or {
        "selected_support": 0,
        "correct_support": 0,
        "observed_metric": None,
        "wilson_lower_bound": 0.0,
        "coverage": 0.0,
    }
    return {
        "enabled": selected_attempt is not None,
        "disabled_reason": disabled_reason,
        "threshold": selected_attempt["threshold"] if selected_attempt else None,
        "direction": direction,
        "comparison": ">=" if direction == "upper" else "<=",
        "metric": metric,
        "target": target,
        "confidence": confidence,
        "minimum_predicted_support": minimum_support,
        "supervised_support": total,
        "reference_support": reference_support,
        "selected_support": metrics["selected_support"],
        "correct_support": metrics["correct_support"],
        "observed_metric": metrics["observed_metric"],
        "wilson_lower_bound": metrics["wilson_lower_bound"],
        "coverage": metrics["coverage"],
        "coverage_unit": "supervised_calibration_rows",
        "best_attempt": best_attempt if selected_attempt is None else None,
    }


def binary_nll_at_temperature(logits: Any, labels: Any, temperature: float) -> float:
    torch = training.torch
    assert torch is not None
    return float(
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float().reshape(-1) / float(temperature),
            labels.float().reshape(-1),
        ).detach().cpu().item()
    )


def fit_binary_temperature(logits: Any, labels: Any) -> dict[str, Any]:
    torch = training.torch
    assert torch is not None
    logits = logits.float().reshape(-1)
    labels = labels.float().reshape(-1)
    if logits.numel() == 0:
        raise ValueError("Cannot fit a calibrator without supervised examples.")
    baseline = binary_nll_at_temperature(logits, labels, 1.0)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100)

    def closure() -> Any:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 10.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / temperature, labels
        )
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError as exc:
        raise RuntimeError("Binary temperature optimization failed.") from exc
    fitted = float(torch.exp(log_temperature).clamp(0.05, 10.0).detach().cpu().item())
    fitted_nll = binary_nll_at_temperature(logits, labels, fitted)
    if fitted_nll <= baseline:
        temperature, final_nll, status = fitted, fitted_nll, "optimized"
    else:
        temperature, final_nll, status = 1.0, baseline, "identity_selected_no_nll_improvement"
    return {
        "method": "temperature_scaling",
        "input_kind": "binary_logit",
        "output_kind": "positive_class_probability",
        "parameters": {"temperature": temperature},
        "fit_status": status,
        "nll_before": baseline,
        "nll_after": final_nll,
    }


def apply_binary_calibrator(logits: Any, calibrator: Mapping[str, Any]) -> Any:
    """Apply sigmoid exactly once to raw binary logits."""

    torch = training.torch
    assert torch is not None
    if calibrator.get("method") != "temperature_scaling" or calibrator.get(
        "input_kind"
    ) != "binary_logit":
        raise ValueError("Unsupported binary calibrator contract.")
    temperature = float(calibrator["parameters"]["temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Calibration temperature must be finite and positive.")
    return torch.sigmoid(logits.float().reshape(-1) / temperature)


def binary_calibration_diagnostics(
    logits: Any, labels: Any, calibrator: Mapping[str, Any], *, bins: int = DEFAULT_DIAGNOSTIC_BINS
) -> dict[str, Any]:
    torch = training.torch
    assert torch is not None
    logits = logits.float().reshape(-1)
    labels = labels.float().reshape(-1)
    raw = torch.sigmoid(logits)
    calibrated = apply_binary_calibrator(logits, calibrator)

    def summarize(probabilities: Any) -> dict[str, Any]:
        reliability = []
        ece = 0.0
        total = int(labels.numel())
        for index in range(bins):
            lower, upper = index / bins, (index + 1) / bins
            mask = (
                probabilities.ge(lower) & probabilities.le(upper)
                if index == bins - 1
                else probabilities.ge(lower) & probabilities.lt(upper)
            )
            count = int(mask.sum().item())
            mean_probability = float(probabilities[mask].mean().item()) if count else None
            empirical = float(labels[mask].mean().item()) if count else None
            if count:
                ece += count / total * abs(float(mean_probability) - float(empirical))
            reliability.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": count,
                    "mean_predicted_probability": mean_probability,
                    "empirical_frequency": empirical,
                }
            )
        return {
            "brier_score": float(torch.mean((probabilities - labels) ** 2).item()),
            "expected_calibration_error": ece,
            "reliability_bins": reliability,
        }

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "bin_count": bins,
        "supervised_rows": int(labels.numel()),
        "positive_support": int(labels.sum().item()),
        "negative_support": int(labels.numel() - labels.sum().item()),
        "raw": summarize(raw),
        "calibrated": summarize(calibrated),
    }


def collect_calibration_logits(
    model: Any, bias_head: Any, opinion_style_head: Any,
    calibration_df: Any, batch_size: int, device: Any,
) -> tuple[Any, Any, Any, Any]:
    torch = training.torch
    assert torch is not None
    loader = training.DataLoader(
        training.BiasOpinionStyleDataset(calibration_df),
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=training.collate_opinion_style_batch,
    )
    bias_logits, style_logits, bias_labels, style_labels = [], [], [], []
    model.eval(); bias_head.eval(); opinion_style_head.eval()
    with torch.no_grad():
        for batch in loader:
            embeddings = training.sentence_embeddings(model, batch["texts"], device)
            bias_logits.append(bias_head(embeddings).detach().cpu().reshape(-1))
            style_logits.append(opinion_style_head(embeddings).detach().cpu().reshape(-1))
            bias_labels.append(batch["bias_labels"].detach().cpu())
            style_labels.append(batch["opinion_style_labels"].detach().cpu())
    return (
        torch.cat(bias_logits), torch.cat(style_logits),
        torch.cat(bias_labels), torch.cat(style_labels),
    )


def _calibrate_head(logits: Any, labels: Any) -> tuple[dict[str, Any], list[float], list[int]]:
    supervised = labels.ne(training.LABEL_IGNORE_INDEX)
    if not bool(supervised.any().item()):
        raise ValueError("Cannot calibrate a head without supervised calibration examples.")
    logits, labels = logits[supervised], labels[supervised]
    calibrator = fit_binary_temperature(logits, labels)
    calibrator.update(
        {
            "schema_version": CALIBRATOR_SCHEMA_VERSION,
            "sample_support": {
                "supervised": int(labels.numel()),
                "positive": int(labels.eq(1).sum().item()),
                "negative": int(labels.eq(0).sum().item()),
            },
            "diagnostics": binary_calibration_diagnostics(logits, labels, calibrator),
        }
    )
    probabilities = [float(value) for value in apply_binary_calibrator(logits, calibrator).tolist()]
    truth = [int(value) for value in labels.tolist()]
    return calibrator, probabilities, truth


def validate_calibration_split_manifest(manifest: Mapping[str, Any]) -> None:
    data_pipeline.validate_split_manifest(manifest)
    if manifest.get("split_strategy") != "group":
        raise data_pipeline.ManifestValidationError(
            "Calibration requires split_strategy='group'."
        )
    if manifest.get("leakage_policy") != "error":
        raise data_pipeline.ManifestValidationError(
            "Calibration requires leakage_policy='error'."
        )
    if not any(
        str(entry.get("partition")) == "calibration"
        for entry in manifest.get("assignments", [])
    ):
        raise data_pipeline.ManifestValidationError(
            "Split manifest has no dedicated calibration assignments."
        )


def _is_missing(value: Any) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    pandas = getattr(training, "pd", None)
    return bool(pandas.isna(value)) if pandas is not None else False


def calibration_data_fingerprint(
    calibration_df: Any, *, partition_id: str,
    split_partition_assignment_sha256: str,
) -> dict[str, Any]:
    records = []
    for _, row in calibration_df.sort_values("record_id", kind="mergesort").iterrows():
        records.append(
            {
                "record_id": str(row["record_id"]),
                "canonical_group_id": str(row["canonical_group_id"]),
                "event_id": None if _is_missing(row.get("event_id")) else str(row.get("event_id")),
                "text_hash": str(row["text_hash"]),
                "bias_label_id": None if _is_missing(row.get("bias_label_id")) else int(row["bias_label_id"]),
                "opinion_style_label_id": None
                if _is_missing(row.get("opinion_style_label_id"))
                else int(row["opinion_style_label_id"]),
            }
        )
    return {
        "schema_version": CALIBRATION_DATA_SCHEMA_VERSION,
        "partition_id": partition_id,
        "record_count": len(records),
        "records_sha256": data_pipeline.canonical_json_sha256(records),
        "split_partition_assignment_sha256": split_partition_assignment_sha256,
    }


def _load_calibration_partition(args: argparse.Namespace) -> dict[str, Any]:
    if args.split_manifest is None:
        raise SystemExit("--split-manifest is required; validation reuse is prohibited.")
    records, canonical_manifest, _stats, bias_map, style_map = training.load_canonical_dataset(args)
    data_hash = canonical_manifest["canonical_data_manifest_sha256"]
    manifest = data_pipeline.load_split_manifest(args.split_manifest)
    validate_calibration_split_manifest(manifest)
    data_pipeline.validate_split_manifest(
        manifest, expected_canonical_data_manifest_sha256=data_hash
    )
    if args.canonical_data_manifest is None:
        raise SystemExit("--canonical-data-manifest is required for calibration binding.")
    stored = json.loads(args.canonical_data_manifest.read_text(encoding="utf-8"))
    data_pipeline.validate_canonical_data_manifest(
        stored, expected_canonical_data_manifest_sha256=data_hash
    )
    calibration_df = data_pipeline.records_for_partition(records, manifest, "calibration")
    if calibration_df.empty:
        raise SystemExit("Split manifest has no dedicated calibration rows.")
    return {
        "calibration_df": calibration_df,
        "bias_label2id": bias_map,
        "opinion_style_label2id": style_map,
        "split_manifest": manifest,
        "canonical_data_manifest_sha256": data_hash,
        "calibration_partition_id": "calibration",
        "split_partition_assignment_sha256": data_pipeline.split_partition_assignment_sha256(
            manifest, "calibration"
        ),
    }


def _artifact_entry_path(artifact: Mapping[str, Any], key: str) -> Path:
    try:
        relative = Path(str(artifact["core"]["files"][key]["path"]))
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"Artifact manifest is missing its {key} entry.") from exc
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(f"Artifact {key} path escapes the project root.") from exc
    return path


def validate_artifact_for_calibration(
    artifact: Mapping[str, Any], *, args: argparse.Namespace,
    model_dir: Path, heads_path: Path, checkpoint: Mapping[str, Any],
    bias_label2id: Mapping[str, int], opinion_style_label2id: Mapping[str, int],
    partition_context: Mapping[str, Any],
) -> None:
    if checkpoint.get("checkpoint_schema_version") != training.CLASSIFICATION_HEADS_SCHEMA_VERSION:
        raise SystemExit("Calibration requires classification_heads/v3.")
    if checkpoint.get("classifier_input_contract") != data_pipeline.CLASSIFIER_INPUT_CONTRACT:
        raise SystemExit("Checkpoint classifier input contract is incompatible.")
    expected_types = {
        "classifier": training.INDEPENDENT_BINARY_HEADS_TYPE,
        "bias": training.BINARY_BIAS_HEAD_TYPE,
        "opinion_style": training.BINARY_OPINION_STYLE_HEAD_TYPE,
    }
    observed_types = {
        "classifier": checkpoint.get("classifier_head_type"),
        "bias": checkpoint.get("bias_head_type"),
        "opinion_style": checkpoint.get("opinion_style_head_type"),
    }
    if observed_types != expected_types:
        raise SystemExit("Checkpoint does not declare the required production head types.")
    core = artifact.get("core", {})
    if _artifact_entry_path(artifact, "model") != model_dir.resolve():
        raise SystemExit("Artifact model path does not match --sbert-model-dir.")
    if _artifact_entry_path(artifact, "classification_heads") != heads_path.resolve():
        raise SystemExit("Artifact heads path does not match the loaded checkpoint.")
    if _artifact_entry_path(artifact, "split_manifest") != Path(args.split_manifest).resolve():
        raise SystemExit("Artifact split manifest path does not match --split-manifest.")
    if _artifact_entry_path(artifact, "canonical_data_manifest") != Path(
        args.canonical_data_manifest
    ).resolve():
        raise SystemExit("Artifact canonical-data manifest path mismatch.")
    mappings = core.get("label_mappings", {})
    if mappings.get("bias_label2id") != dict(bias_label2id):
        raise SystemExit("Artifact bias label mapping does not match the checkpoint.")
    if mappings.get("opinion_style_label2id") != dict(opinion_style_label2id):
        raise SystemExit("Artifact opinion-style label mapping does not match the checkpoint.")
    if core.get("head_type") != training.INDEPENDENT_BINARY_HEADS_TYPE:
        raise SystemExit("Artifact classifier head type is not production binary v1.")
    if partition_context["split_manifest"].get("canonical_data_manifest_sha256") != partition_context[
        "canonical_data_manifest_sha256"
    ]:
        raise SystemExit("Split and canonical-data manifests are not bound.")


def _decision_state_coverage(
    scores: Sequence[float], lower: Mapping[str, Any], upper: Mapping[str, Any],
    names: tuple[str, str, str],
) -> dict[str, Any]:
    counts = {name: 0 for name in names}
    for score in scores:
        if lower.get("enabled") and float(score) <= float(lower["threshold"]):
            state = names[0]
        elif upper.get("enabled") and float(score) >= float(upper["threshold"]):
            state = names[2]
        else:
            state = names[1]
        counts[state] += 1
    total = len(scores)
    return {
        name: {"count": count, "percentage": count / total if total else 0.0}
        for name, count in counts.items()
    }


def _validate_threshold_order(lower: Mapping[str, Any], upper: Mapping[str, Any]) -> bool:
    return not (
        lower.get("enabled")
        and upper.get("enabled")
        and float(lower["threshold"]) >= float(upper["threshold"])
    )


def _region_summary(name: str, region: Mapping[str, Any]) -> str:
    if not region["enabled"]:
        return f"{name}=disabled({region['disabled_reason']})"
    return (
        f"{name}={region['threshold']:.6f} support={region['selected_support']} "
        f"metric={region['observed_metric']:.4f} lcb={region['wilson_lower_bound']:.4f} "
        f"coverage={region['coverage']:.4f}"
    )


def main() -> None:
    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
    args = parse_args()
    apply_metadata_defaults(args)
    for name in (
        "bias_target_precision", "bias_target_npv",
        "opinion_target_precision", "opinion_target_npv",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between zero and one.")
    if args.minimum_predicted_support <= 0:
        raise SystemExit("--minimum-predicted-support must be positive.")
    if not 0.0 < args.precision_confidence < 1.0:
        raise SystemExit("--precision-confidence must be strictly between zero and one.")

    training.import_training_dependencies()
    torch = training.torch
    assert torch is not None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    context = _load_calibration_partition(args)
    calibration_df = context["calibration_df"]
    heads_path = Path(args.classification_heads) if args.classification_heads else args.sbert_model_dir / "classification_heads.pt"
    checkpoint = torch.load(heads_path, map_location=device)
    model, bias_head, style_head, bias_map, style_map = infer_bias_llm.load_sbert_and_heads(
        args.sbert_model_dir, heads_path, device, compatibility_mode="strict"
    )
    if context["bias_label2id"] != bias_map or context["opinion_style_label2id"] != style_map:
        raise SystemExit("Calibration data label mappings do not match the checkpoint.")
    if not args.artifact_manifest.exists():
        raise SystemExit("A production calibration requires artifact_manifest.json.")
    artifact = artifact_manifest.load_artifact_manifest(
        args.artifact_manifest, artifact_root=PROJECT_ROOT, verify_files=True
    )
    validate_artifact_for_calibration(
        artifact, args=args, model_dir=args.sbert_model_dir, heads_path=heads_path,
        checkpoint=checkpoint, bias_label2id=bias_map,
        opinion_style_label2id=style_map, partition_context=context,
    )
    bias_logits, style_logits, bias_labels, style_labels = collect_calibration_logits(
        model, bias_head, style_head, calibration_df, args.batch_size, device
    )
    bias_calibrator, bias_scores, bias_truth = _calibrate_head(bias_logits, bias_labels)
    style_calibrator, style_scores, style_truth = _calibrate_head(style_logits, style_labels)
    regions = {
        "bias_no_clear_max": select_confident_region(
            bias_scores, bias_truth, direction="lower", target=args.bias_target_npv,
            minimum_support=args.minimum_predicted_support, confidence=args.precision_confidence,
        ),
        "bias_clear_min": select_confident_region(
            bias_scores, bias_truth, direction="upper", target=args.bias_target_precision,
            minimum_support=args.minimum_predicted_support, confidence=args.precision_confidence,
        ),
        "opinion_objective_max": select_confident_region(
            style_scores, style_truth, direction="lower", target=args.opinion_target_npv,
            minimum_support=args.minimum_predicted_support, confidence=args.precision_confidence,
        ),
        "opinion_opinionated_min": select_confident_region(
            style_scores, style_truth, direction="upper", target=args.opinion_target_precision,
            minimum_support=args.minimum_predicted_support, confidence=args.precision_confidence,
        ),
    }
    order_valid = _validate_threshold_order(
        regions["bias_no_clear_max"], regions["bias_clear_min"]
    ) and _validate_threshold_order(
        regions["opinion_objective_max"], regions["opinion_opinionated_min"]
    )
    files = artifact["core"]["files"]
    calibration_data = calibration_data_fingerprint(
        calibration_df, partition_id="calibration",
        split_partition_assignment_sha256=context["split_partition_assignment_sha256"],
    )
    binding = {
        "artifact_manifest_sha256": artifact["artifact_manifest_sha256"],
        "model_sha256": files["model"]["sha256"],
        "heads_sha256": files["classification_heads"]["sha256"],
        "model_artifact_path": files["model"]["path"],
        "classification_heads_path": files["classification_heads"]["path"],
        "split_manifest_sha256": context["split_manifest"]["split_content_sha256"],
        "split_manifest_file_sha256": files["split_manifest"]["sha256"],
        "canonical_data_manifest_sha256": context["canonical_data_manifest_sha256"],
        "canonical_data_manifest_file_sha256": files["canonical_data_manifest"]["sha256"],
        "calibration_partition_id": "calibration",
        "calibration_records_sha256": calibration_data["records_sha256"],
        "calibration_partition_assignment_sha256": calibration_data["split_partition_assignment_sha256"],
        "artifact_code_sha256": artifact_manifest.canonical_json_sha256(files["code"]),
        "calibration_code_sha256": artifact_manifest.sha256_path(Path(__file__).resolve()),
        "checkpoint_schema_version": training.CLASSIFICATION_HEADS_SCHEMA_VERSION,
        "classifier_input_contract": data_pipeline.CLASSIFIER_INPUT_CONTRACT,
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "decision_policy_version": DECISION_POLICY_VERSION,
    }
    head_types = {
        "classifier": training.INDEPENDENT_BINARY_HEADS_TYPE,
        "bias": training.BINARY_BIAS_HEAD_TYPE,
        "opinion_style": training.BINARY_OPINION_STYLE_HEAD_TYPE,
    }
    output = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "decision_policy_version": DECISION_POLICY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "calibration_status": "valid" if order_valid else "invalid_threshold_order",
        "artifact_binding": binding,
        "calibration_data": calibration_data,
        "head_types": head_types,
        "label_mappings": {
            "bias_label2id": bias_map,
            "opinion_style_label2id": style_map,
            "opinion_style_target_mapping": checkpoint["opinion_style_target_mapping"],
        },
        "classifier_input_contract": data_pipeline.CLASSIFIER_INPUT_CONTRACT,
        "calibrators": {"bias": bias_calibrator, "opinion_style": style_calibrator},
        "thresholds": regions,
        "decision_state_coverage": {
            "bias": _decision_state_coverage(
                bias_scores, regions["bias_no_clear_max"], regions["bias_clear_min"],
                ("no_clear_bias", "possible_bias", "clear_bias"),
            ),
            "opinion_style": _decision_state_coverage(
                style_scores, regions["opinion_objective_max"], regions["opinion_opinionated_min"],
                ("objective_style", "uncertain", "opinionated_style"),
            ),
        },
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Calibration rows=%d schema=%s", len(calibration_df), CALIBRATION_SCHEMA_VERSION)
    LOGGER.info("Bias T=%.6f | %s | %s", bias_calibrator["parameters"]["temperature"], _region_summary("no_clear", regions["bias_no_clear_max"]), _region_summary("clear", regions["bias_clear_min"]))
    LOGGER.info("Opinion T=%.6f | %s | %s", style_calibrator["parameters"]["temperature"], _region_summary("objective", regions["opinion_objective_max"]), _region_summary("opinionated", regions["opinion_opinionated_min"]))
    LOGGER.info("Diagnostics: %s", args.output_file)
    if not order_valid:
        raise SystemExit("Independent threshold searches produced a non-strict pair; calibration was not bound.")
    bound = artifact_manifest.bind_calibration(
        artifact, artifact_root=PROJECT_ROOT, calibration_path=args.output_file
    )
    artifact_manifest.write_artifact_manifest(args.artifact_manifest, bound)


if __name__ == "__main__":
    main()
