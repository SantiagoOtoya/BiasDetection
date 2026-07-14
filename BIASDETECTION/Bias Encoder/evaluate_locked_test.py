#!/usr/bin/env python
"""Evaluate one frozen classifier on the locked test exactly once."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
LLM_DIR = PROJECT_ROOT / "LLM-inference"
for import_path in (PROJECT_ROOT, LLM_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import artifact_manifest
import calibrate_sbert_heads as calibration
import data_pipeline
import evaluation
import finetune_all_mpnet_babe as training
import infer_bias_llm as inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen v3 artifact on locked_test exactly once."
    )
    parser.add_argument("--sbert-model-dir", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--canonical-data-manifest", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--evaluation-lock", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--promote-if-gates-pass", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must contain a JSON object: {path}")
    return value


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _validate_paths(args: argparse.Namespace) -> None:
    required = (
        args.sbert_model_dir,
        args.calibration_file,
        args.artifact_manifest,
        args.data_dir,
        args.split_manifest,
        args.canonical_data_manifest,
        args.gate_config,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Locked-test preflight is missing required paths: {missing}")
    if args.output_file.exists() or args.evaluation_lock.exists():
        raise SystemExit("Locked-test output or one-shot marker already exists.")
    for path in (args.output_file, args.evaluation_lock):
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise SystemExit(
                f"Locked-test outputs must stay under {PROJECT_ROOT}: {path}"
            ) from exc
    if args.bootstrap_resamples <= 0 or args.batch_size <= 0:
        raise SystemExit("Bootstrap resamples and batch size must be positive.")
    if not args.promote_if_gates_pass and not args.preflight_only:
        raise SystemExit(
            "A one-shot evaluation requires --promote-if-gates-pass so a passing "
            "artifact is promoted atomically."
        )


def _metadata_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "classification_heads": args.sbert_model_dir / "classification_heads.pt",
            "output_file": args.calibration_file,
            "precision_confidence": None,
            "train_data_glob": None,
            "test_data_glob": None,
            "data_glob": None,
            "split_column": None,
            "train_split_value": None,
            "test_split_value": None,
            "text_column": None,
            "bias_label_column": None,
            "opinion_label_column": None,
            "include_no_agreement": None,
            "missing_label_policy": None,
            "no_dedupe": False,
            "seed": None,
            "group_column": None,
            "article_id_column": None,
            "event_column": None,
        }
    )
    result = argparse.Namespace(**values)
    calibration.apply_metadata_defaults(result)
    return result


def _create_lock(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise SystemExit(f"Locked test has already been opened: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_locked_partition(
    data_args: argparse.Namespace, split_manifest_path: Path
) -> tuple[Any, dict[str, Any], dict[str, int], dict[str, int]]:
    """Load canonical records, then expose only the locked-test partition."""

    records, canonical, _stats, bias_map, style_map = training.load_canonical_dataset(
        data_args
    )
    split = data_pipeline.load_split_manifest(split_manifest_path)
    data_pipeline.validate_split_manifest(
        split,
        expected_canonical_data_manifest_sha256=canonical[
            "canonical_data_manifest_sha256"
        ],
    )
    locked = data_pipeline.records_for_partition(records, split, "locked_test")
    return locked, split, bias_map, style_map


def main() -> None:
    args = parse_args()
    for name in (
        "sbert_model_dir", "calibration_file", "artifact_manifest", "data_dir",
        "split_manifest", "canonical_data_manifest", "gate_config", "output_file",
        "evaluation_lock",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    _validate_paths(args)
    gates = _load_json(args.gate_config, "promotion gate configuration")
    if gates.get("schema_version") != "promotion_gates/v1":
        raise SystemExit("Unsupported promotion gate configuration schema.")
    targets = gates.get("targets")
    if not isinstance(targets, dict):
        raise SystemExit("Promotion gate configuration has no targets object.")

    training.import_training_dependencies()
    torch = training.torch
    assert torch is not None
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available; CPU fallback is forbidden.")
    device = torch.device(args.device)
    heads_path = args.sbert_model_dir / "classification_heads.pt"
    model, bias_head, style_head, bias_map, style_map = inference.load_sbert_and_heads(
        args.sbert_model_dir, heads_path, device, compatibility_mode="strict"
    )
    selection_args = SimpleNamespace(
        selection_mode="calibrated",
        sbert_model_dir=args.sbert_model_dir,
        classification_heads=heads_path,
        calibration_file=args.calibration_file,
        artifact_manifest=args.artifact_manifest,
        bias_threshold=None,
        opinion_threshold=None,
        include_possible_bias=False,
    )
    selection = inference.resolve_selection_config(
        selection_args,
        bias_map,
        style_map,
        getattr(style_head, "head_type", None),
        heads_path=heads_path,
        checkpoint_head_type_declared=getattr(
            style_head, "checkpoint_declared_head_type", None
        ),
        bias_head_type=getattr(bias_head, "head_type", None),
        classifier_head_type=getattr(style_head, "classifier_head_type", None),
        checkpoint_schema_version=getattr(style_head, "checkpoint_schema_version", None),
        opinion_style_target_mapping=getattr(
            style_head, "opinion_style_target_mapping", None
        ),
        classifier_input_contract=getattr(style_head, "classifier_input_contract", None),
        allowed_promotion_states=frozenset({"pending_locked_test"}),
    )
    manifest = artifact_manifest.load_artifact_manifest(
        args.artifact_manifest, artifact_root=PROJECT_ROOT, verify_files=True
    )
    files = manifest["core"]["files"]
    declared_split = (PROJECT_ROOT / files["split_manifest"]["path"]).resolve()
    declared_canonical = (
        PROJECT_ROOT / files["canonical_data_manifest"]["path"]
    ).resolve()
    if declared_split != args.split_manifest:
        raise SystemExit("CLI split manifest does not match the frozen artifact.")
    if declared_canonical != args.canonical_data_manifest:
        raise SystemExit("CLI canonical-data manifest does not match the frozen artifact.")
    frozen = manifest.get("frozen_artifact")
    if not isinstance(frozen, dict) or not frozen.get("frozen_artifact_sha256"):
        raise SystemExit("Artifact is not frozen for locked-test evaluation.")
    if args.preflight_only:
        print("Locked-test preflight passed; no partition rows were loaded.")
        return

    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _create_lock(
        args.evaluation_lock,
        {
            "schema_version": "locked_test_once/v1",
            "status": "started",
            "started_at_utc": started_at,
            "frozen_artifact_sha256": frozen["frozen_artifact_sha256"],
        },
    )

    data_args = _metadata_args(args)
    locked, split, observed_bias_map, observed_style_map = _load_locked_partition(
        data_args, args.split_manifest
    )
    if observed_bias_map != bias_map or observed_style_map != style_map:
        raise SystemExit("Locked-test data label mappings do not match the checkpoint.")
    if locked.empty:
        raise SystemExit("The locked_test partition is empty.")
    predictions = inference.classify_sentences(
        locked["text"].astype(str).tolist(),
        model,
        bias_head,
        style_head,
        bias_map,
        style_map,
        device,
        args.batch_size,
        selection,
    )

    bias_truth: list[int] = []
    bias_scores: list[float] = []
    bias_states: list[str] = []
    style_truth: list[int] = []
    style_scores: list[float] = []
    style_states: list[str] = []
    style_groups: list[str] = []
    for (_, row), prediction in zip(locked.iterrows(), predictions):
        if training.pd.notna(row["bias_label_id"]):
            bias_truth.append(int(row["bias_label_id"]))
            bias_scores.append(float(prediction.p_bias))
            bias_states.append(str(prediction.bias_assessment))
        if training.pd.notna(row["opinion_style_label_id"]):
            style_truth.append(int(row["opinion_style_label_id"]))
            style_scores.append(float(prediction.p_opinionated_style))
            style_states.append(str(prediction.opinion_style))
            style_groups.append(str(row["canonical_group_id"]))

    scorecard = evaluation.evaluate_binary_acceptance_gates(
        bias_truth=bias_truth,
        bias_probabilities=bias_scores,
        bias_states=bias_states,
        opinion_style_truth=style_truth,
        opinion_style_probabilities=style_scores,
        opinion_style_states=style_states,
        group_ids=style_groups,
        confidence=float(gates.get("confidence", 0.95)),
        bootstrap_resamples=args.bootstrap_resamples,
        seed=int(gates.get("bootstrap_seed", 42)),
        targets=targets,
    )
    scorecard.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "frozen_artifact_sha256": frozen["frozen_artifact_sha256"],
            "gate_config_sha256": artifact_manifest.sha256_path(args.gate_config),
            "locked_test_partition_assignment_sha256": (
                data_pipeline.split_partition_assignment_sha256(split, "locked_test")
            ),
            "locked_test_rows": int(len(locked)),
        }
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    data_pipeline.write_json(args.output_file, scorecard)
    bound = artifact_manifest.bind_locked_test_evaluation(
        manifest,
        artifact_root=PROJECT_ROOT,
        scorecard_path=args.output_file,
        release_gate_results=scorecard,
    )
    artifact_manifest.write_artifact_manifest(args.artifact_manifest, bound)
    data_pipeline.write_json(
        args.evaluation_lock,
        {
            "schema_version": "locked_test_once/v1",
            "status": "complete",
            "started_at_utc": started_at,
            "completed_at_utc": scorecard["created_at_utc"],
            "frozen_artifact_sha256": frozen["frozen_artifact_sha256"],
            "scorecard_sha256": artifact_manifest.sha256_path(args.output_file),
            "promotion_state": scorecard["acceptance_gate"]["promotion_state"],
        },
    )
    print(json.dumps(scorecard, indent=2, sort_keys=True))
    if not scorecard["acceptance_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
