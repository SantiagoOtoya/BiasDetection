#!/usr/bin/env python
"""Fine-tune all-mpnet-base-v2 SBERT on BABE media-bias labels.

The default setup uses BABE_HF when present, otherwise the curated BABE final
label CSV files in ``data``. Production training retains independently labeled
rows, masks invalid task targets, freezes MPNet encoder layers 0-5, and
fine-tunes the last six encoder layers with two scalar binary MLP heads:

    total_loss = bias_loss + weight * opinion_style_loss

Both losses are computed in the same forward pass and update the shared SBERT
encoder plus both heads with one backward pass.

Example:
    python "Bias Encoder/finetune_all_mpnet_babe.py" --epochs 4 --batch-size 16 --fp16
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import artifact_manifest
import data_pipeline
import evaluation as foundation_evaluation


LOGGER = logging.getLogger("finetune_all_mpnet_babe")
SUPPORTED_DATA_SUFFIXES = {".csv", ".parquet"}
np = None
pd = None
torch = None
SentenceTransformer = None
DataLoader = None
WeightedRandomSampler = None


LEGACY_FLAT_HEAD_TYPE = "legacy_flat_v1"
HIERARCHICAL_ORDINAL_HEAD_TYPE = "hierarchical_ordinal_v1"
PRODUCTION_CLASSIFIER_MODE = "production"
LEGACY_HIERARCHICAL_CLASSIFIER_MODE = "legacy-hierarchical"
LEGACY_FLAT_CLASSIFIER_MODE = "legacy-flat"
CLASSIFICATION_HEADS_SCHEMA_VERSION = "classification_heads/v3"
LEGACY_CLASSIFICATION_HEADS_SCHEMA_VERSION = "classification_heads/v2"
BINARY_BIAS_HEAD_TYPE = "binary_bias_logit_v1"
BINARY_OPINION_STYLE_HEAD_TYPE = "binary_opinion_style_logit_v1"
INDEPENDENT_BINARY_HEADS_TYPE = "independent_binary_heads_v1"
LABEL_IGNORE_INDEX = -100
CANONICAL_OPINION_LABELS = data_pipeline.CANONICAL_OPINION_LABELS
CANONICAL_OPINION_LABEL2ID = data_pipeline.CANONICAL_OPINION_LABEL2ID
OPINION_STYLE_LABELS = data_pipeline.OPINION_STYLE_LABELS
OPINION_STYLE_LABEL2ID = data_pipeline.OPINION_STYLE_LABEL2ID


class CheckpointCompatibilityError(ValueError):
    """Raised when a classification-head checkpoint cannot be loaded safely."""


def resolve_classifier_mode(args: argparse.Namespace) -> str:
    """Resolve the production mode or an explicitly requested legacy alias."""

    requested = getattr(args, "classifier_mode", None)
    legacy_head_type = getattr(args, "opinion_head_type", None)
    legacy_mapping = {
        HIERARCHICAL_ORDINAL_HEAD_TYPE: LEGACY_HIERARCHICAL_CLASSIFIER_MODE,
        LEGACY_FLAT_HEAD_TYPE: LEGACY_FLAT_CLASSIFIER_MODE,
    }
    if legacy_head_type is not None:
        legacy_mode = legacy_mapping.get(str(legacy_head_type))
        if legacy_mode is None:
            raise CheckpointCompatibilityError(
                f"Unsupported legacy opinion head type: {legacy_head_type!r}."
            )
        if requested is not None and requested != legacy_mode:
            raise CheckpointCompatibilityError(
                "--classifier-mode conflicts with the deprecated "
                "--opinion-head-type compatibility alias."
            )
        requested = legacy_mode
    mode = str(requested or PRODUCTION_CLASSIFIER_MODE)
    allowed = {
        PRODUCTION_CLASSIFIER_MODE,
        LEGACY_HIERARCHICAL_CLASSIFIER_MODE,
        LEGACY_FLAT_CLASSIFIER_MODE,
    }
    if mode not in allowed:
        raise CheckpointCompatibilityError(
            f"Unsupported classifier mode {mode!r}; expected one of {sorted(allowed)}."
        )
    args.classifier_mode = mode
    if mode == LEGACY_HIERARCHICAL_CLASSIFIER_MODE:
        args.opinion_head_type = HIERARCHICAL_ORDINAL_HEAD_TYPE
    elif mode == LEGACY_FLAT_CLASSIFIER_MODE:
        args.opinion_head_type = LEGACY_FLAT_HEAD_TYPE
    else:
        args.opinion_head_type = None
    return mode


def import_training_dependencies() -> None:
    global DataLoader, SentenceTransformer, WeightedRandomSampler, np, pd, torch

    missing: list[str] = []
    try:
        import numpy as numpy_module
    except ImportError:
        missing.append("numpy")
    else:
        np = numpy_module

    try:
        import pandas as pandas_module
    except ImportError:
        missing.append("pandas")
    else:
        pd = pandas_module

    try:
        import torch as torch_module
    except ImportError:
        missing.append("torch")
    else:
        torch = torch_module

    try:
        from sentence_transformers import SentenceTransformer as sentence_transformer_class
    except ImportError:
        missing.append("sentence-transformers")
    else:
        SentenceTransformer = sentence_transformer_class

    try:
        from torch.utils.data import (
            DataLoader as data_loader_class,
            WeightedRandomSampler as weighted_random_sampler_class,
        )
    except ImportError:
        if "torch" not in missing:
            missing.append("torch")
    else:
        DataLoader = data_loader_class
        WeightedRandomSampler = weighted_random_sampler_class

    if missing:
        unique_missing = ", ".join(sorted(set(missing)))
        raise SystemExit(
            "Missing required fine-tuning package(s): "
            f"{unique_missing}\nInstall them with:\n"
            '  python -m pip install -r "Bias Encoder/requirements-finetune.txt"'
        )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir / "BABE_HF"
    if not default_data_dir.exists():
        default_data_dir = script_dir / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune sentence-transformers/all-mpnet-base-v2 on BABE "
            "bias and opinion labels while freezing the lower MPNet layers."
        )
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Base SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing BABE CSV or parquet files.",
    )
    parser.add_argument(
        "--data-glob",
        nargs="+",
        default=["final_labels_*.csv"],
        help=(
            "One or more glob patterns, resolved under --data-dir. Used for "
            "unsplit data, or with --split-column. CSV and parquet are supported."
        ),
    )
    parser.add_argument(
        "--train-data-glob",
        nargs="+",
        default=None,
        help=(
            "Glob pattern(s) for an existing BABE train split. When set, "
            "validation is split only from these rows."
        ),
    )
    parser.add_argument(
        "--test-data-glob",
        nargs="+",
        default=None,
        help="Glob pattern(s) for an existing BABE test split kept out of training.",
    )
    parser.add_argument(
        "--split-column",
        default=None,
        help=(
            "Optional column containing split names. If set, --data-glob is "
            "loaded and rows matching --train-split-value are split into "
            "train/validation while --test-split-value rows are held out."
        ),
    )
    parser.add_argument(
        "--train-split-value",
        default="train",
        help="Value in --split-column that identifies train rows.",
    )
    parser.add_argument(
        "--test-split-value",
        default="test",
        help="Value in --split-column that identifies held-out test rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "models" / "all-mpnet-base-v2-babe",
        help="Directory where the fine-tuned SBERT model and metadata are saved.",
    )
    parser.add_argument("--text-column", default="text", help="Input text column.")
    parser.add_argument(
        "--bias-label-column",
        default="label_bias",
        help="Column used for the primary bias classification loss.",
    )
    parser.add_argument(
        "--opinion-label-column",
        default="label_opinion",
        help="Source opinion-tier column mapped to the binary style target.",
    )
    parser.add_argument(
        "--include-no-agreement",
        action="store_true",
        help=(
            "Legacy-only row policy. Production always retains the row and "
            "masks an invalid task target independently."
        ),
    )
    parser.add_argument(
        "--missing-label-policy",
        choices=["mask", "drop"],
        default="mask",
        help=(
            "Mask a task loss when its label is absent (default), or retain the "
            "legacy behavior of dropping rows missing either task label."
        ),
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable global text deduplication only in explicit legacy split mode.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of total optimizer steps used for linear LR warmup.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay / L2 regularization strength.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.15,
        help="Development fraction. Set to 0 to skip development metrics.",
    )
    parser.add_argument(
        "--calibration-size",
        type=float,
        default=0.15,
        help="Dedicated calibration fraction; never reuse development silently.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Locked-test fraction for group-aware splitting.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["group", "legacy"],
        default="group",
        help="Use group-disjoint splitting by default; legacy preserves row-level compatibility.",
    )
    parser.add_argument(
        "--group-column",
        default="news_link",
        help="Article URL/group column used to derive canonical article grouping.",
    )
    parser.add_argument(
        "--article-id-column",
        default=None,
        help="Optional alias for --group-column when the source uses another article ID field.",
    )
    parser.add_argument(
        "--event-column",
        default=None,
        help="Optional event/story column; BASIL event/story IDs are always respected when present.",
    )
    parser.add_argument(
        "--leakage-policy",
        choices=["error", "warn"],
        default="error",
        help="Fail or explicitly warn when article/event/text leakage is detected.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Path for immutable split_manifest/v1; defaults under --output-dir.",
    )
    parser.add_argument(
        "--canonical-data-manifest",
        type=Path,
        default=None,
        help="Path for canonical_data_manifest/v1; defaults beside the split manifest.",
    )
    parser.add_argument(
        "--precision-confidence",
        type=float,
        default=0.95,
        help="One-sided confidence level recorded for later calibration and gates.",
    )
    parser.add_argument(
        "--allow-legacy-calibration",
        action="store_true",
        help="Permit an explicit legacy run without a dedicated calibration partition.",
    )
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument(
        "--freeze-first-n-layers",
        type=int,
        default=6,
        help=(
            "Number of lower MPNet encoder layers to freeze. MPNet base has "
            "12 layers, so the default fine-tunes only the last 6 layers."
        ),
    )
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Also freeze MPNet token and position embeddings.",
    )
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=256,
        help="Hidden size for each one-hidden-layer MLP classification head.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability used inside both classification heads.",
    )
    parser.add_argument(
        "--classifier-mode",
        choices=[
            PRODUCTION_CLASSIFIER_MODE,
            LEGACY_HIERARCHICAL_CLASSIFIER_MODE,
            LEGACY_FLAT_CLASSIFIER_MODE,
        ],
        default=None,
        help=(
            "Production uses independent scalar bias and opinion-style heads. "
            "The other modes are explicit legacy/experimental compatibility paths."
        ),
    )
    parser.add_argument(
        "--opinion-head-type",
        choices=[HIERARCHICAL_ORDINAL_HEAD_TYPE, LEGACY_FLAT_HEAD_TYPE],
        default=None,
        help=(
            "Deprecated compatibility alias. Supplying this flag explicitly "
            "selects the matching legacy classifier mode."
        ),
    )
    parser.add_argument(
        "--middle-class-weight",
        type=float,
        default=1.0,
        help="Additional per-example weight for the middle opinion tier in hierarchical loss.",
    )
    parser.add_argument(
        "--opinion-focal-gamma",
        type=float,
        default=0.0,
        help="Optional focal-loss gamma for hierarchical BCE terms; 0 preserves ordinary BCE.",
    )
    parser.add_argument(
        "--opinion-style-loss-weight",
        "--opinion-loss-alpha",
        dest="opinion_style_loss_weight",
        type=float,
        default=0.3,
        help=(
            "Opinion-style loss weight in bias_loss + weight * "
            "opinion_style_loss. The old option name is retained as an alias."
        ),
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default="balanced",
        help=(
            "Use inverse-frequency class weights for each production binary "
            "task. Legacy hierarchical runs may use their older policy."
        ),
    )
    parser.add_argument(
        "--opinion-sampling",
        choices=["none", "balanced"],
        default="none",
        help=(
            "Legacy-hierarchical deterministic tier sampling for the training "
            "partition. Production mode does not use this option."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Optional torch device.")
    parser.add_argument("--fp16", action="store_true", help="Use AMP mixed precision.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set to 0 to disable clipping.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation metrics after each epoch.",
    )
    parser.add_argument(
        "--evaluate-test-final",
        action="store_true",
        help="Evaluate the held-out test split once after training finishes.",
    )
    parser.add_argument(
        "--save-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the best validation checkpoint under output-dir/best.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help=(
            "Stop after this many non-improving validation epochs. Set to 0 "
            "to disable early stopping."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=["multi_objective", "monitor_metric"],
        default="multi_objective",
        help=(
            "Use the mode-appropriate ordered development objectives, or an "
            "explicit single-metric experimental/legacy selector."
        ),
    )
    parser.add_argument(
        "--monitor-metric",
        default="bias_macro_f1",
        choices=[
            "loss",
            "bias_accuracy",
            "bias_balanced_accuracy",
            "bias_macro_f1",
            "bias_biased_precision",
            "bias_biased_recall",
            "opinion_style_accuracy",
            "opinion_style_balanced_accuracy",
            "opinion_style_macro_f1",
            "opinion_style_objective_recall",
            "opinion_style_opinionated_recall",
            "opinion_accuracy",
            "opinion_macro_f1",
            "opinion_middle_recall",
        ],
        help="Validation metric used for best-checkpoint selection.",
    )
    args = parser.parse_args()
    resolve_classifier_mode(args)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_files(data_dir: Path, patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matched = sorted(data_dir.glob(pattern))
        if not matched:
            LOGGER.warning("No files matched %s under %s", pattern, data_dir)
        files.extend(matched)

    unique_files = sorted({path.resolve() for path in files})
    if not unique_files:
        raise FileNotFoundError(
            f"No training files found in {data_dir} with patterns {list(patterns)}"
        )
    return unique_files


def read_babe_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    elif suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                f"Reading parquet data requires pyarrow or fastparquet: {path}"
            ) from exc
    else:
        raise ValueError(
            f"Unsupported BABE data file type {path.suffix!r}: {path}. "
            f"Supported suffixes: {', '.join(sorted(SUPPORTED_DATA_SUFFIXES))}"
        )
    df["source_file"] = path.name
    return df


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_label(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_bias_label(series: pd.Series) -> pd.Series:
    labels = normalize_label(series)
    label_aliases = {
        "0": "Non-biased",
        "0.0": "Non-biased",
        "nonbiased": "Non-biased",
        "non-biased": "Non-biased",
        "1": "Biased",
        "1.0": "Biased",
        "biased": "Biased",
    }
    return labels.map(lambda label: label_aliases.get(label.casefold(), label))


def ordered_label_names(labels: Iterable[str]) -> list[str]:
    labels = list(labels)
    preferred = [
        "Non-biased",
        "Biased",
        "Entirely factual",
        "Somewhat factual but also opinionated",
        "Expresses writer's opinion",
        "Expresses writer’s opinion",
        "No agreement",
    ]
    ordered = [label for label in preferred if label in labels]
    ordered.extend(sorted(label for label in labels if label not in set(ordered)))
    return ordered


def _is_missing_scalar(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _raw_opinion_label_summary(
    frame: pd.DataFrame, requested_column: str
) -> dict[str, int]:
    column = requested_column
    if column not in frame.columns and "label_opinion" in frame.columns:
        column = "label_opinion"
    if column not in frame.columns:
        return {"<missing column>": int(len(frame))}

    counts: Counter[str] = Counter()
    for value in frame[column].tolist():
        label = "<missing>" if _is_missing_scalar(value) else str(value)
        counts[label] += 1
    return dict(sorted(counts.items()))


def _canonical_opinion_label_summary(
    frame: pd.DataFrame,
    *,
    label_id_column: str = "opinion_label",
) -> dict[str, int]:
    counts: Counter[str] = Counter({label: 0 for label in CANONICAL_OPINION_LABELS})
    counts.update({"<missing>": 0, "<unrecognized>": 0, "<No agreement>": 0})
    names = frame.get("opinion_label_name", pd.Series([None] * len(frame)))
    identifiers = frame.get(label_id_column, pd.Series([None] * len(frame)))

    for name, identifier in zip(names.tolist(), identifiers.tolist()):
        if not _is_missing_scalar(identifier):
            label_id = int(identifier)
            if 0 <= label_id < len(CANONICAL_OPINION_LABELS):
                counts[CANONICAL_OPINION_LABELS[label_id]] += 1
                continue
        if _is_missing_scalar(name) or not str(name).strip():
            counts["<missing>"] += 1
        elif str(name).casefold().startswith("no agreement"):
            counts["<No agreement>"] += 1
        else:
            counts["<unrecognized>"] += 1
    return dict(counts)


def _opinion_style_label_summary(
    frame: pd.DataFrame,
    *,
    label_id_column: str = "opinion_style_label",
) -> dict[str, int]:
    counts: Counter[str] = Counter({label: 0 for label in OPINION_STYLE_LABELS})
    counts["<masked>"] = 0
    identifiers = frame.get(label_id_column, pd.Series([None] * len(frame)))
    for identifier in identifiers.tolist():
        if _is_missing_scalar(identifier):
            counts["<masked>"] += 1
            continue
        label_id = int(identifier)
        if 0 <= label_id < len(OPINION_STYLE_LABELS):
            counts[OPINION_STYLE_LABELS[label_id]] += 1
        else:
            counts["<masked>"] += 1
    return dict(counts)


def _partition_opinion_diagnostics(
    partitions: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, object]]:
    diagnostics: dict[str, dict[str, object]] = {}
    for partition in data_pipeline.PARTITIONS:
        subset = partitions[partition]
        diagnostics[partition] = {
            "record_count": int(len(subset)),
            "group_count": int(subset["canonical_group_id"].nunique(dropna=True)),
            "opinion_counts": _canonical_opinion_label_summary(
                subset, label_id_column="opinion_label_id"
            ),
        }
    return diagnostics


def _format_partition_opinion_diagnostics(
    diagnostics: Mapping[str, Mapping[str, object]],
) -> str:
    return "; ".join(
        (
            f"{partition}: groups={diagnostics[partition]['group_count']}, "
            f"opinions={diagnostics[partition]['opinion_counts']}"
        )
        for partition in data_pipeline.PARTITIONS
    )


def read_babe_files(files: list[Path]) -> pd.DataFrame:
    frames = [read_babe_file(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def find_split_files(data_dir: Path, split_name: str) -> list[Path]:
    files: list[Path] = []
    for suffix in SUPPORTED_DATA_SUFFIXES:
        files.extend(data_dir.glob(f"*{split_name}*{suffix}"))
    return sorted({path.resolve() for path in files})


def resolve_input_column(
    df: pd.DataFrame,
    requested_column: str,
    role: str,
    aliases: Iterable[str] = (),
) -> str:
    if requested_column in df.columns:
        return requested_column

    for alias in aliases:
        if alias in df.columns:
            LOGGER.info(
                "Using %s column %r because requested column %r was not found",
                role,
                alias,
                requested_column,
            )
            return alias

    candidates = [requested_column, *aliases]
    raise KeyError(
        f"Missing required {role} column. Tried: {', '.join(candidates)}. "
        f"Available columns: {', '.join(df.columns)}"
    )


def load_raw_dataset(args: argparse.Namespace) -> pd.DataFrame:
    if args.train_data_glob or args.test_data_glob:
        if not args.train_data_glob:
            raise ValueError("--test-data-glob requires --train-data-glob.")

        train_files = resolve_data_files(args.data_dir, args.train_data_glob)
        LOGGER.info(
            "Loading %d train file(s): %s",
            len(train_files),
            ", ".join(path.name for path in train_files),
        )
        train_df = read_babe_files(train_files)
        train_df["dataset_split"] = "train"

        frames = [train_df]
        if args.test_data_glob:
            test_files = resolve_data_files(args.data_dir, args.test_data_glob)
            LOGGER.info(
                "Loading %d test file(s): %s",
                len(test_files),
                ", ".join(path.name for path in test_files),
            )
            test_df = read_babe_files(test_files)
            test_df["dataset_split"] = "test"
            frames.append(test_df)

        return pd.concat(frames, ignore_index=True)

    if args.split_column:
        files = resolve_data_files(args.data_dir, args.data_glob)
        LOGGER.info(
            "Loading %d split-column file(s): %s",
            len(files),
            ", ".join(path.name for path in files),
        )
        df = read_babe_files(files)
        if args.split_column not in df.columns:
            raise KeyError(f"Missing split column: {args.split_column}")

        split_values = normalize_label(df[args.split_column]).str.casefold()
        train_value = args.train_split_value.casefold()
        test_value = args.test_split_value.casefold()
        train_mask = split_values == train_value
        test_mask = split_values == test_value

        ignored_rows = len(df) - int(train_mask.sum()) - int(test_mask.sum())
        if ignored_rows:
            LOGGER.warning(
                "Ignoring %d row(s) whose %s value is neither %r nor %r",
                ignored_rows,
                args.split_column,
                args.train_split_value,
                args.test_split_value,
            )

        train_df = df[train_mask].copy()
        train_df["dataset_split"] = "train"
        test_df = df[test_mask].copy()
        test_df["dataset_split"] = "test"
        return pd.concat([train_df, test_df], ignore_index=True)

    auto_train_files = find_split_files(args.data_dir, "train")
    auto_test_files = find_split_files(args.data_dir, "test")
    if auto_train_files:
        LOGGER.info(
            "Auto-detected %d train file(s): %s",
            len(auto_train_files),
            ", ".join(path.name for path in auto_train_files),
        )
        train_df = read_babe_files(auto_train_files)
        train_df["dataset_split"] = "train"
        frames = [train_df]
        if auto_test_files:
            LOGGER.info(
                "Auto-detected %d test file(s): %s",
                len(auto_test_files),
                ", ".join(path.name for path in auto_test_files),
            )
            test_df = read_babe_files(auto_test_files)
            test_df["dataset_split"] = "test"
            frames.append(test_df)
        return pd.concat(frames, ignore_index=True)

    files = resolve_data_files(args.data_dir, args.data_glob)
    LOGGER.info(
        "Loading %d unsplit data file(s): %s",
        len(files),
        ", ".join(path.name for path in files),
    )
    df = read_babe_files(files)
    df["dataset_split"] = "train"
    return df


def load_canonical_dataset(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, int], dict[str, int]]:
    """Load all source rows into canonical records before any partitioning."""

    raw = load_raw_dataset(args)
    LOGGER.info(
        "Opinion labels raw input: %s",
        _raw_opinion_label_summary(raw, args.opinion_label_column),
    )
    source_frames: list[pd.DataFrame] = []
    mbic_frames: list[pd.DataFrame] = []
    if "source_file" not in raw.columns:
        raw["source_file"] = "<in_memory>"
    for source_file, source_frame in raw.groupby("source_file", sort=True, dropna=False):
        source_dataset = data_pipeline.source_dataset_for_file(str(source_file))
        canonical = data_pipeline.canonicalize_records(
            source_frame.copy(),
            source_dataset=source_dataset,
            text_column=args.text_column,
            bias_label_column=args.bias_label_column,
            opinion_label_column=args.opinion_label_column,
            group_column=args.group_column,
            article_id_column=args.article_id_column,
            event_column=args.event_column,
        )
        if source_dataset == "MBIC":
            mbic_frames.append(canonical)
        else:
            source_frames.append(canonical)
    if not source_frames:
        raise ValueError("No non-MBIC source rows were available for canonicalization.")
    canonical_records = pd.concat(source_frames, ignore_index=True)
    mbic_stats: dict[str, Any] = {"mbic_matched": 0, "mbic_unmatched": 0}
    if mbic_frames:
        canonical_records = data_pipeline.merge_mbic_annotations(
            canonical_records,
            pd.concat(mbic_frames, ignore_index=True),
            mbic_stats,
        )
    LOGGER.info(
        "Opinion labels after canonical normalization: %s",
        _canonical_opinion_label_summary(canonical_records),
    )
    missing_label_policy = getattr(args, "missing_label_policy", "mask")
    if missing_label_policy not in {"mask", "drop"}:
        raise ValueError(
            "--missing-label-policy must be 'mask' or 'drop', "
            f"got {missing_label_policy!r}."
        )
    classifier_mode = resolve_classifier_mode(args)
    production_mode = classifier_mode == PRODUCTION_CLASSIFIER_MODE
    opinion_head_type = getattr(args, "opinion_head_type", None)
    if not production_mode:
        validate_opinion_head_type(opinion_head_type)
    cleaned, cleaning_stats = data_pipeline.clean_and_dedupe_records(
        canonical_records,
        include_no_agreement=(True if production_mode else args.include_no_agreement),
        require_both_labels=missing_label_policy == "drop",
        global_dedupe=not args.no_dedupe,
        opinion_target_column=(
            "opinion_style_label" if production_mode else "opinion_label"
        ),
    )
    cleaning_stats.update(mbic_stats)
    if production_mode:
        LOGGER.info(
            "Opinion-style targets after normalization/masking/dedupe: %s",
            _opinion_style_label_summary(cleaned),
        )
    else:
        LOGGER.info(
            "Legacy opinion tiers after missing/no-agreement policy and dedupe: %s",
            _canonical_opinion_label_summary(cleaned),
        )
    if cleaned.empty:
        raise ValueError("No labeled canonical records remain after cleaning.")
    bias_labels = ordered_label_names(
        cleaned.loc[cleaned["bias_label"].notna(), "bias_label_name"].dropna().unique()
    )
    if len(bias_labels) < 2:
        raise ValueError("Training requires at least two bias labels after cleaning.")
    bias_label2id = {label: index for index, label in enumerate(bias_labels)}
    if production_mode:
        opinion_label2id = dict(OPINION_STYLE_LABEL2ID)
    elif opinion_head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE:
        observed_opinion_ids = {
            int(value) for value in cleaned["opinion_label"].dropna().tolist()
        }
        missing_opinion_ids = sorted(
            set(CANONICAL_OPINION_LABEL2ID.values()).difference(observed_opinion_ids)
        )
        if missing_opinion_ids:
            missing_opinion_labels = [
                CANONICAL_OPINION_LABELS[index] for index in missing_opinion_ids
            ]
            raise ValueError(
                "Hierarchical opinion training requires all three canonical tiers; "
                f"missing {missing_opinion_labels}."
            )
        opinion_label2id = dict(CANONICAL_OPINION_LABEL2ID)
    else:
        opinion_labels = ordered_label_names(
            cleaned.loc[
                cleaned["opinion_label"].notna(), "opinion_label_name"
            ].dropna().unique()
        )
        if len(opinion_labels) < 2:
            raise ValueError("Training requires at least two opinion labels after cleaning.")
        opinion_label2id = {label: index for index, label in enumerate(opinion_labels)}
    cleaned["bias_label_id"] = (
        cleaned["bias_label_name"].map(bias_label2id).astype("Int64")
    )
    if production_mode:
        cleaned["opinion_style_label_id"] = cleaned[
            "opinion_style_label"
        ].astype("Int64")
    elif opinion_head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE:
        cleaned["opinion_label_id"] = cleaned["opinion_label"].astype("Int64")
    else:
        recognized_opinion_names = cleaned["opinion_label_name"].where(
            cleaned["opinion_label"].notna()
        )
        cleaned["opinion_label_id"] = recognized_opinion_names.map(
            opinion_label2id
        ).astype("Int64")
    source_files = [
        args.data_dir / str(name)
        for name in sorted({str(value) for value in raw["source_file"].dropna().tolist()})
        if (args.data_dir / str(name)).exists()
    ]
    canonical_data_manifest = data_pipeline.build_canonical_data_manifest(
        cleaned,
        cleaning_stats,
        source_files=source_files,
    )
    return cleaned, canonical_data_manifest, cleaning_stats, bias_label2id, opinion_label2id


def _split_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    split_path = args.split_manifest or (args.output_dir / "split_manifest.json")
    data_manifest_path = args.canonical_data_manifest or split_path.with_name(
        "canonical_data_manifest.json"
    )
    return split_path, data_manifest_path


def _write_or_validate_canonical_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_hash = existing.get("canonical_data_manifest_sha256")
        if existing_hash != manifest.get("canonical_data_manifest_sha256"):
            raise data_pipeline.ManifestValidationError(
                f"Canonical data manifest is immutable and does not match current data: {path}"
            )
        return
    data_pipeline.write_json(path, manifest)


def _legacy_row_assignments(
    records: pd.DataFrame, args: argparse.Namespace
) -> dict[str, str]:
    """Create an explicit row-level compatibility split without reusing dev."""

    source_test = records[records["input_split"].astype(str).str.casefold() == "test"].copy()
    pool = records[records["input_split"].astype(str).str.casefold() != "test"].copy()
    if pool.empty:
        raise ValueError("Legacy splitting requires at least one non-test row.")
    assignments: dict[str, str] = {}
    if source_test.empty:
        pool, source_test = split_dataset(pool, args.test_size, args.seed)
    for record_id in source_test["record_id"].tolist():
        assignments[str(record_id)] = "locked_test"
    if args.calibration_size > 0:
        relative_calibration = min(0.95, args.calibration_size / max(0.01, 1.0 - args.test_size))
        pool, calibration = split_dataset(pool, relative_calibration, args.seed + 1)
        for record_id in calibration["record_id"].tolist():
            assignments[str(record_id)] = "calibration"
    elif not args.allow_legacy_calibration:
        raise ValueError(
            "--calibration-size 0 requires --allow-legacy-calibration."
        )
    if args.validation_size > 0:
        remaining_fraction = max(0.01, 1.0 - args.test_size - args.calibration_size)
        relative_development = min(0.95, args.validation_size / remaining_fraction)
        train, development = split_dataset(pool, relative_development, args.seed + 2)
    else:
        train = pool.sample(frac=1.0, random_state=args.seed + 2).reset_index(drop=True)
        development = pool.iloc[0:0].copy()
    for record_id in development["record_id"].tolist():
        assignments[str(record_id)] = "development"
    for record_id in train["record_id"].tolist():
        assignments[str(record_id)] = "train"
    return assignments


def prepare_data_and_splits(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare canonical data plus immutable four-way partition manifests."""

    classifier_mode = resolve_classifier_mode(args)
    production_mode = classifier_mode == PRODUCTION_CLASSIFIER_MODE
    if not 0.0 < args.precision_confidence < 1.0:
        raise ValueError("--precision-confidence must be strictly between zero and one.")
    if args.calibration_size <= 0 and not args.allow_legacy_calibration:
        raise ValueError("--calibration-size 0 requires --allow-legacy-calibration.")
    if args.split_strategy == "group" and args.no_dedupe:
        raise ValueError("--no-dedupe is only available with --split-strategy legacy.")
    if args.split_strategy == "group" and args.leakage_policy != "error":
        raise ValueError("Group-aware splitting requires --leakage-policy error.")
    if args.middle_class_weight <= 0:
        raise ValueError("--middle-class-weight must be positive.")
    if args.opinion_focal_gamma < 0:
        raise ValueError("--opinion-focal-gamma must be non-negative.")
    if args.opinion_style_loss_weight < 0:
        raise ValueError("--opinion-style-loss-weight must be non-negative.")
    if production_mode:
        if args.missing_label_policy != "mask":
            raise ValueError(
                "Production mode requires --missing-label-policy mask so one "
                "task cannot discard valid supervision for the other."
            )
        if args.include_no_agreement:
            raise ValueError(
                "--include-no-agreement is a legacy-only option; production "
                "always retains rows and masks invalid task targets independently."
            )
        if (
            args.middle_class_weight != 1.0
            or args.opinion_focal_gamma != 0.0
            or args.opinion_sampling != "none"
        ):
            raise ValueError(
                "Middle-tier weighting, hierarchical focal loss, and opinion-tier "
                "sampling require --classifier-mode legacy-hierarchical."
            )
    else:
        validate_opinion_head_type(args.opinion_head_type)
    if args.checkpoint_selection == "monitor_metric":
        production_metrics = {
            "loss",
            "bias_accuracy",
            "bias_balanced_accuracy",
            "bias_macro_f1",
            "bias_biased_precision",
            "bias_biased_recall",
            "opinion_style_accuracy",
            "opinion_style_balanced_accuracy",
            "opinion_style_macro_f1",
            "opinion_style_objective_recall",
            "opinion_style_opinionated_recall",
        }
        legacy_metrics = {
            "loss",
            "bias_accuracy",
            "bias_macro_f1",
            "opinion_accuracy",
            "opinion_macro_f1",
            "opinion_middle_recall",
        }
        allowed_metrics = production_metrics if production_mode else legacy_metrics
        if args.monitor_metric not in allowed_metrics:
            raise ValueError(
                f"--monitor-metric {args.monitor_metric!r} is not available in "
                f"--classifier-mode {classifier_mode}."
            )
    if (
        args.opinion_sampling == "balanced"
        and classifier_mode != LEGACY_HIERARCHICAL_CLASSIFIER_MODE
    ):
        raise ValueError(
            "--opinion-sampling balanced is supported only by "
            f"{HIERARCHICAL_ORDINAL_HEAD_TYPE}."
        )
    records, canonical_data_manifest, cleaning_stats, bias_label2id, opinion_label2id = (
        load_canonical_dataset(args)
    )
    split_path, data_manifest_path = _split_paths(args)
    _write_or_validate_canonical_manifest(data_manifest_path, canonical_data_manifest)
    data_hash = str(canonical_data_manifest["canonical_data_manifest_sha256"])
    if split_path.exists():
        split_manifest = data_pipeline.load_split_manifest(split_path)
        data_pipeline.validate_split_manifest(
            split_manifest,
            expected_canonical_data_manifest_sha256=data_hash,
        )
        assignment = {
            str(entry["record_id"]): str(entry["partition"])
            for entry in split_manifest["assignments"]
        }
        assigned = records.copy()
        assigned["partition"] = assigned["record_id"].map(assignment)
        assigned["split_unit_id"] = assigned["record_id"].map(
            {
                str(entry["record_id"]): str(entry["split_unit_id"])
                for entry in split_manifest["assignments"]
            }
        )
        if assigned["partition"].isna().any():
            raise data_pipeline.ManifestValidationError(
                "Existing split manifest does not assign every current canonical record."
            )
    elif args.split_strategy == "group":
        assigned, split_manifest = data_pipeline.build_group_split_manifest(
            records,
            seed=args.seed,
            validation_size=args.validation_size,
            calibration_size=args.calibration_size,
            test_size=args.test_size,
            canonical_data_manifest_sha256=data_hash,
            cleaning_stats=cleaning_stats,
            leakage_policy=args.leakage_policy,
        )
        data_pipeline.write_json(split_path, split_manifest)
    else:
        assignments = _legacy_row_assignments(records, args)
        assigned, split_manifest = data_pipeline.build_legacy_split_manifest(
            records,
            assignments,
            seed=args.seed,
            canonical_data_manifest_sha256=data_hash,
            cleaning_stats=cleaning_stats,
        )
        data_pipeline.write_json(split_path, split_manifest)
    partitions = {
        partition: assigned[assigned["partition"] == partition].reset_index(drop=True)
        for partition in data_pipeline.PARTITIONS
    }
    if production_mode:
        partition_diagnostics = {
            partition: {
                "record_count": int(len(partitions[partition])),
                "group_count": int(
                    partitions[partition]["canonical_group_id"].nunique(dropna=True)
                ),
                "opinion_counts": _opinion_style_label_summary(
                    partitions[partition],
                    label_id_column="opinion_style_label_id",
                ),
            }
            for partition in data_pipeline.PARTITIONS
        }
    else:
        partition_diagnostics = _partition_opinion_diagnostics(partitions)
    for partition in data_pipeline.PARTITIONS:
        diagnostic = partition_diagnostics[partition]
        LOGGER.info(
            "%s %s partition: records=%d groups=%d counts=%s",
            "Opinion-style targets" if production_mode else "Legacy opinion tiers",
            partition,
            diagnostic["record_count"],
            diagnostic["group_count"],
            diagnostic["opinion_counts"],
        )
    partition_diagnostics_text = _format_partition_opinion_diagnostics(
        partition_diagnostics
    )
    if partitions["train"].empty:
        raise ValueError(
            "Training partition is empty. Partition diagnostics: "
            f"{partition_diagnostics_text}"
        )
    if args.validation_size > 0 and partitions["development"].empty:
        raise ValueError(
            "Development partition is empty; adjust split fractions or data. "
            f"Partition diagnostics: {partition_diagnostics_text}"
        )
    if args.calibration_size > 0 and partitions["calibration"].empty:
        raise ValueError(
            "Calibration partition is empty; adjust split fractions or data. "
            f"Partition diagnostics: {partition_diagnostics_text}"
        )
    if production_mode:
        train_labels = set(
            partitions["train"]["opinion_style_label_id"]
            .dropna()
            .astype(int)
            .tolist()
        )
        missing_train_ids = sorted(
            set(OPINION_STYLE_LABEL2ID.values()).difference(train_labels)
        )
        if missing_train_ids:
            missing_names = [OPINION_STYLE_LABELS[index] for index in missing_train_ids]
            raise ValueError(
                "Production opinion-style training requires both binary targets "
                f"in the train partition; missing {missing_names}. Partition "
                f"diagnostics: {partition_diagnostics_text}"
            )
    elif args.opinion_head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE:
        train_labels = set(
            partitions["train"]["opinion_label_id"].dropna().astype(int).tolist()
        )
        missing_train_ids = sorted(
            set(CANONICAL_OPINION_LABEL2ID.values()).difference(train_labels)
        )
        if missing_train_ids:
            missing_names = [CANONICAL_OPINION_LABELS[index] for index in missing_train_ids]
            raise ValueError(
                "Hierarchical opinion training requires every tier in the train "
                f"partition; missing {missing_names}. Partition diagnostics: "
                f"{partition_diagnostics_text}"
            )
    prepared = {
        "records": assigned,
        "partitions": partitions,
        "split_manifest": split_manifest,
        "split_manifest_path": split_path,
        "canonical_data_manifest": canonical_data_manifest,
        "canonical_data_manifest_path": data_manifest_path,
        "cleaning_stats": cleaning_stats,
        "bias_label2id": bias_label2id,
        "opinion_target_label2id": opinion_label2id,
        "classifier_mode": classifier_mode,
    }
    if production_mode:
        prepared["opinion_style_label2id"] = opinion_label2id
    else:
        prepared["opinion_label2id"] = opinion_label2id
    return prepared


def load_dataset(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
    df = load_raw_dataset(args)

    text_column = resolve_input_column(df, args.text_column, "text")
    bias_label_column = resolve_input_column(
        df,
        args.bias_label_column,
        "bias label",
        aliases=("label",),
    )
    opinion_label_column = resolve_input_column(
        df,
        args.opinion_label_column,
        "opinion label",
    )

    df["text"] = normalize_text(df[text_column])
    df["bias_label_name"] = normalize_bias_label(df[bias_label_column])
    df["opinion_label_name"] = normalize_label(df[opinion_label_column])
    df = df[
        (df["text"] != "")
        & (df["bias_label_name"] != "")
        & (df["opinion_label_name"] != "")
    ].copy()

    if not args.include_no_agreement:
        before = len(df)
        no_agreement_mask = (
            df["bias_label_name"].str.casefold().str.startswith("no agreement")
            | df["opinion_label_name"].str.casefold().str.startswith("no agreement")
        )
        df = df[~no_agreement_mask].copy()
        LOGGER.info("Dropped %d row(s) with 'No agreement' labels", before - len(df))

    if not args.no_dedupe:
        before = len(df)
        label_counts_by_text = df.groupby(["dataset_split", "text"])[
            ["bias_label_name", "opinion_label_name"]
        ].nunique()
        conflicts = label_counts_by_text[
            (label_counts_by_text["bias_label_name"] > 1)
            | (label_counts_by_text["opinion_label_name"] > 1)
        ].reset_index()[["dataset_split", "text"]]
        if not conflicts.empty:
            df = df.merge(
                conflicts.assign(_conflicting_label=True),
                on=["dataset_split", "text"],
                how="left",
            )
            df = df[df["_conflicting_label"].isna()].drop(
                columns=["_conflicting_label"]
            )
        df = df.drop_duplicates(
            subset=[
                "dataset_split",
                "text",
                "bias_label_name",
                "opinion_label_name",
            ]
        )
        df = df.copy()
        LOGGER.info("Removed %d duplicate/conflicting row(s)", before - len(df))

    train_pool_df = df[df["dataset_split"] == "train"].copy()
    test_df = df[df["dataset_split"] == "test"].copy()
    if train_pool_df.empty:
        raise ValueError("No train rows found after filtering.")

    if not test_df.empty:
        overlap_count = len(set(train_pool_df["text"]).intersection(test_df["text"]))
        if overlap_count:
            LOGGER.warning(
                "%d text(s) appear in both train and test splits; keeping split labels unchanged",
                overlap_count,
            )

    for column in ["bias_label_name", "opinion_label_name"]:
        counts = train_pool_df[column].value_counts()
        rare_labels = set(counts[counts < 2].index)
        if rare_labels:
            LOGGER.warning(
                "Dropping %s label(s) with fewer than 2 train examples: %s",
                column,
                ", ".join(sorted(rare_labels)),
            )
            df = df[~df[column].isin(rare_labels)].copy()
            train_pool_df = df[df["dataset_split"] == "train"].copy()
            test_df = df[df["dataset_split"] == "test"].copy()

    bias_labels = ordered_label_names(train_pool_df["bias_label_name"].unique())
    opinion_labels = ordered_label_names(train_pool_df["opinion_label_name"].unique())
    if len(bias_labels) < 2:
        raise ValueError("Training requires at least two bias labels after filtering.")
    if len(opinion_labels) < 2:
        raise ValueError("Training requires at least two opinion labels after filtering.")

    bias_label2id = {label: idx for idx, label in enumerate(bias_labels)}
    opinion_label2id = {label: idx for idx, label in enumerate(opinion_labels)}
    before_test = len(test_df)
    test_df = test_df[
        test_df["bias_label_name"].isin(bias_label2id)
        & test_df["opinion_label_name"].isin(opinion_label2id)
    ].copy()
    if before_test - len(test_df):
        LOGGER.warning(
            "Dropped %d test row(s) with labels not present in the train split",
            before_test - len(test_df),
        )

    train_pool_df["bias_label_id"] = (
        train_pool_df["bias_label_name"].map(bias_label2id).astype(int)
    )
    train_pool_df["opinion_label_id"] = (
        train_pool_df["opinion_label_name"].map(opinion_label2id).astype(int)
    )
    if not test_df.empty:
        test_df["bias_label_id"] = test_df["bias_label_name"].map(bias_label2id).astype(int)
        test_df["opinion_label_id"] = (
            test_df["opinion_label_name"].map(opinion_label2id).astype(int)
        )
    else:
        test_df["bias_label_id"] = pd.Series(dtype=int)
        test_df["opinion_label_id"] = pd.Series(dtype=int)

    LOGGER.info(
        "Loaded split rows: train_pool=%d test=%d",
        len(train_pool_df),
        len(test_df),
    )
    for label_name, count in train_pool_df["bias_label_name"].value_counts().sort_index().items():
        LOGGER.info("Bias label    %-45s %6d", label_name, count)
    for label_name, count in train_pool_df["opinion_label_name"].value_counts().sort_index().items():
        LOGGER.info("Opinion label %-45s %6d", label_name, count)

    return (
        train_pool_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        bias_label2id,
        opinion_label2id,
    )


def split_dataset(
    df: pd.DataFrame, validation_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if validation_size <= 0:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True), df.iloc[0:0]

    opinion_target_column = (
        "opinion_style_label_id"
        if "opinion_style_label_id" in df.columns
        else "opinion_label_id"
    )
    stratify_key = (
        df["bias_label_id"].astype(str)
        + "::"
        + df[opinion_target_column].astype(str)
    )
    if stratify_key.value_counts().min() < 2:
        LOGGER.warning("Using bias-only stratification because some joint labels are rare")
        stratify_key = df["bias_label_id"]

    try:
        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            df,
            test_size=validation_size,
            random_state=seed,
            stratify=stratify_key,
        )
    except Exception as exc:  # pragma: no cover - fallback for small custom datasets
        LOGGER.warning("Falling back to non-stratified split: %s", exc)
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_count = max(1, int(round(len(shuffled) * validation_size)))
        val_df = shuffled.iloc[:val_count]
        train_df = shuffled.iloc[val_count:]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class BiasOpinionDataset:
    """Legacy three-tier dataset retained for calibration and experiments."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.texts = df["text"].tolist()
        self.bias_labels = [
            int(value) if pd.notna(value) else LABEL_IGNORE_INDEX
            for value in df["bias_label_id"].tolist()
        ]
        self.opinion_labels = [
            int(value) if pd.notna(value) else LABEL_IGNORE_INDEX
            for value in df["opinion_label_id"].tolist()
        ]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "text": self.texts[index],
            "bias_label": self.bias_labels[index],
            "opinion_label": self.opinion_labels[index],
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "texts": [str(item["text"]) for item in batch],
        "bias_labels": torch.tensor(
            [int(item["bias_label"]) for item in batch], dtype=torch.long
        ),
        "opinion_labels": torch.tensor(
            [int(item["opinion_label"]) for item in batch], dtype=torch.long
        ),
    }


def freeze_mpnet_layers(
    model: SentenceTransformer, first_n_layers: int, freeze_embeddings: bool
) -> dict[str, int]:
    transformer = model._first_module()
    auto_model = getattr(transformer, "auto_model", None)
    if auto_model is None:
        raise ValueError("Could not find the underlying Hugging Face model.")

    encoder = getattr(auto_model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise ValueError("Expected an MPNet-like model with encoder.layer modules.")

    if first_n_layers < 0 or first_n_layers > len(layers):
        raise ValueError(
            f"--freeze-first-n-layers must be between 0 and {len(layers)}"
        )

    for layer_idx, layer in enumerate(layers):
        requires_grad = layer_idx >= first_n_layers
        for parameter in layer.parameters():
            parameter.requires_grad = requires_grad

    if freeze_embeddings and hasattr(auto_model, "embeddings"):
        for parameter in auto_model.embeddings.parameters():
            parameter.requires_grad = False

    return parameter_counts(model)


def parameter_counts(model: SentenceTransformer) -> dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_mlp_head(
    embedding_dim: int, hidden_dim: int, num_labels: int, dropout: float
) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(embedding_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, num_labels),
    )


def build_binary_head(
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    *,
    head_type: str,
) -> torch.nn.Module:
    if head_type not in {BINARY_BIAS_HEAD_TYPE, BINARY_OPINION_STYLE_HEAD_TYPE}:
        raise CheckpointCompatibilityError(
            f"Unsupported binary head type: {head_type!r}."
        )
    head = build_mlp_head(embedding_dim, hidden_dim, 1, dropout)
    head.head_type = head_type
    return head


def binary_head_architecture(
    *,
    head_type: str,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    logit_name: str,
) -> dict[str, Any]:
    if head_type not in {BINARY_BIAS_HEAD_TYPE, BINARY_OPINION_STYLE_HEAD_TYPE}:
        raise CheckpointCompatibilityError(
            f"Unsupported binary head type: {head_type!r}."
        )
    return {
        "implementation": "one_hidden_layer_mlp",
        "head_type": head_type,
        "embedding_dim": int(embedding_dim),
        "head_hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "output_dim": 1,
        "logit_names": [logit_name],
        "probability_transform": "sigmoid",
    }


class BiasOpinionStyleDataset:
    def __init__(self, df: pd.DataFrame) -> None:
        self.texts = [
            data_pipeline.serialize_classifier_input(value)
            for value in df["text"].tolist()
        ]
        self.bias_labels = [
            int(value) if pd.notna(value) else LABEL_IGNORE_INDEX
            for value in df["bias_label_id"].tolist()
        ]
        self.opinion_style_labels = [
            int(value) if pd.notna(value) else LABEL_IGNORE_INDEX
            for value in df["opinion_style_label_id"].tolist()
        ]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "text": self.texts[index],
            "bias_label": self.bias_labels[index],
            "opinion_style_label": self.opinion_style_labels[index],
        }


def collate_opinion_style_batch(
    batch: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "texts": [str(item["text"]) for item in batch],
        "bias_labels": torch.tensor(
            [int(item["bias_label"]) for item in batch], dtype=torch.long
        ),
        "opinion_style_labels": torch.tensor(
            [int(item["opinion_style_label"]) for item in batch], dtype=torch.long
        ),
    }


def validate_opinion_head_type(head_type: str) -> str:
    if head_type not in {LEGACY_FLAT_HEAD_TYPE, HIERARCHICAL_ORDINAL_HEAD_TYPE}:
        raise CheckpointCompatibilityError(
            "Unsupported opinion head type "
            f"{head_type!r}; expected {HIERARCHICAL_ORDINAL_HEAD_TYPE!r} or "
            f"{LEGACY_FLAT_HEAD_TYPE!r}."
        )
    return head_type


def validate_hierarchical_opinion_label_mapping(
    opinion_label2id: Mapping[str, int],
) -> None:
    normalized = {str(label): int(index) for label, index in opinion_label2id.items()}
    if normalized != CANONICAL_OPINION_LABEL2ID:
        raise CheckpointCompatibilityError(
            "Hierarchical opinion heads require the canonical label mapping "
            f"{CANONICAL_OPINION_LABEL2ID}, got {normalized}."
        )


def opinion_head_architecture(
    *,
    head_type: str,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    opinion_label2id: Mapping[str, int],
) -> dict[str, Any]:
    validate_opinion_head_type(head_type)
    architecture: dict[str, Any] = {
        "implementation": "one_hidden_layer_mlp",
        "embedding_dim": int(embedding_dim),
        "head_hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "label_order": [
            label for label, _index in sorted(opinion_label2id.items(), key=lambda item: item[1])
        ],
    }
    if head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE:
        validate_hierarchical_opinion_label_mapping(opinion_label2id)
        architecture.update(
            {
                "output_dim": 2,
                "logit_names": [
                    "q_nonfactual_logit",
                    "q_writer_given_nonfactual_logit",
                ],
                "probability_reconstruction": "hierarchical_ordinal/v1",
            }
        )
    else:
        architecture.update(
            {
                "output_dim": len(opinion_label2id),
                "logit_names": "flat_class_logits",
                "probability_reconstruction": "softmax",
            }
        )
    return architecture


def build_opinion_head(
    *,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    opinion_label2id: Mapping[str, int],
    head_type: str,
) -> torch.nn.Module:
    validate_opinion_head_type(head_type)
    if head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE:
        validate_hierarchical_opinion_label_mapping(opinion_label2id)
        num_outputs = 2
    else:
        num_outputs = len(opinion_label2id)
    head = build_mlp_head(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_labels=num_outputs,
        dropout=dropout,
    )
    head.head_type = head_type
    head.opinion_head_architecture = opinion_head_architecture(
        head_type=head_type,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        opinion_label2id=opinion_label2id,
    )
    return head


def _flat_output_dimension(state_dict: Mapping[str, Any]) -> int | None:
    final_weight = state_dict.get("3.weight")
    shape = getattr(final_weight, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return int(shape[0])


def opinion_head_type_from_checkpoint(checkpoint: Mapping[str, Any]) -> str:
    declared = checkpoint.get("head_type")
    state_dict = checkpoint.get("opinion_head_state_dict")
    if not isinstance(state_dict, Mapping):
        raise CheckpointCompatibilityError(
            "Classification-head checkpoint is missing opinion_head_state_dict."
        )
    output_dim = _flat_output_dimension(state_dict)
    if declared is None:
        if output_dim == 3:
            return LEGACY_FLAT_HEAD_TYPE
        raise CheckpointCompatibilityError(
            "Checkpoint has no head_type and does not match the known legacy "
            "three-output opinion-head layout."
        )
    head_type = validate_opinion_head_type(str(declared))
    expected_outputs = 2 if head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE else None
    if expected_outputs is not None and output_dim != expected_outputs:
        raise CheckpointCompatibilityError(
            f"Checkpoint declares {head_type} but its opinion state dict has "
            f"{output_dim!r} output rows; expected {expected_outputs}."
        )
    if head_type == LEGACY_FLAT_HEAD_TYPE and output_dim is None:
        raise CheckpointCompatibilityError(
            "Legacy flat checkpoint lacks the expected final-layer state-dict shape."
        )
    return head_type


def load_opinion_head_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device
) -> tuple[torch.nn.Module, str]:
    try:
        opinion_label2id = {
            str(label): int(index)
            for label, index in dict(checkpoint["opinion_label2id"]).items()
        }
        embedding_dim = int(checkpoint["embedding_dim"])
        hidden_dim = int(checkpoint["head_hidden_dim"])
        dropout = float(checkpoint["dropout"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(
            "Classification-head checkpoint is missing required opinion-head metadata."
        ) from exc
    head_type = opinion_head_type_from_checkpoint(checkpoint)
    if head_type == LEGACY_FLAT_HEAD_TYPE:
        output_dim = _flat_output_dimension(checkpoint["opinion_head_state_dict"])
        if output_dim != len(opinion_label2id):
            raise CheckpointCompatibilityError(
                "Legacy flat checkpoint final-layer width does not match its label mapping."
            )
    head = build_opinion_head(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        opinion_label2id=opinion_label2id,
        head_type=head_type,
    ).to(device)
    try:
        head.load_state_dict(checkpoint["opinion_head_state_dict"])
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"Could not load {head_type} opinion-head state dict: {exc}"
        ) from exc
    head.head_type = head_type
    return head, head_type


def load_classification_heads_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: torch.device,
    *,
    compatibility_mode: str = "strict",
) -> tuple[torch.nn.Module, torch.nn.Module, str]:
    """Load v3 production heads or explicitly opted-in legacy heads."""

    schema = checkpoint.get("checkpoint_schema_version")
    if schema == CLASSIFICATION_HEADS_SCHEMA_VERSION:
        if compatibility_mode != "strict":
            raise CheckpointCompatibilityError(
                "Production v3 checkpoints must be loaded in strict mode."
            )
        if checkpoint.get("classifier_mode") != PRODUCTION_CLASSIFIER_MODE:
            raise CheckpointCompatibilityError(
                "A v3 checkpoint must declare classifier_mode='production'."
            )
        if checkpoint.get("classifier_input_contract") != data_pipeline.CLASSIFIER_INPUT_CONTRACT:
            raise CheckpointCompatibilityError(
                "A v3 checkpoint does not declare the required target-only "
                "classifier input contract."
            )
        if checkpoint.get("bias_head_type") != BINARY_BIAS_HEAD_TYPE or checkpoint.get(
            "opinion_style_head_type"
        ) != BINARY_OPINION_STYLE_HEAD_TYPE:
            raise CheckpointCompatibilityError(
                "A v3 checkpoint does not declare the required binary head types."
            )
        try:
            embedding_dim = int(checkpoint["embedding_dim"])
            hidden_dim = int(checkpoint["head_hidden_dim"])
            dropout = float(checkpoint["dropout"])
            bias_state = checkpoint["bias_head_state_dict"]
            style_state = checkpoint["opinion_style_head_state_dict"]
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                "Production checkpoint is missing required head metadata."
            ) from exc
        if _flat_output_dimension(bias_state) != 1 or _flat_output_dimension(style_state) != 1:
            raise CheckpointCompatibilityError(
                "Production checkpoint heads must each emit exactly one logit."
            )
        bias_head = build_binary_head(
            embedding_dim,
            hidden_dim,
            dropout,
            head_type=BINARY_BIAS_HEAD_TYPE,
        ).to(device)
        style_head = build_binary_head(
            embedding_dim,
            hidden_dim,
            dropout,
            head_type=BINARY_OPINION_STYLE_HEAD_TYPE,
        ).to(device)
        try:
            bias_head.load_state_dict(bias_state)
            style_head.load_state_dict(style_state)
        except RuntimeError as exc:
            raise CheckpointCompatibilityError(
                f"Could not load production binary head state: {exc}"
            ) from exc
        return bias_head, style_head, PRODUCTION_CLASSIFIER_MODE

    if compatibility_mode != "legacy":
        raise CheckpointCompatibilityError(
            "Legacy or untyped classification-head checkpoints require "
            "compatibility_mode='legacy'."
        )
    try:
        bias_label2id = dict(checkpoint["bias_label2id"])
        embedding_dim = int(checkpoint["embedding_dim"])
        hidden_dim = int(checkpoint["head_hidden_dim"])
        dropout = float(checkpoint["dropout"])
        bias_state = checkpoint["bias_head_state_dict"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(
            "Legacy checkpoint is missing required bias-head metadata."
        ) from exc
    bias_head = build_mlp_head(
        embedding_dim, hidden_dim, len(bias_label2id), dropout
    ).to(device)
    try:
        bias_head.load_state_dict(bias_state)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"Could not load legacy bias-head state: {exc}"
        ) from exc
    opinion_head, head_type = load_opinion_head_from_checkpoint(checkpoint, device)
    legacy_mode = (
        LEGACY_HIERARCHICAL_CLASSIFIER_MODE
        if head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE
        else LEGACY_FLAT_CLASSIFIER_MODE
    )
    return bias_head, opinion_head, legacy_mode


def reconstruct_hierarchical_opinion_probabilities(opinion_outputs: torch.Tensor) -> torch.Tensor:
    if opinion_outputs.ndim < 1 or opinion_outputs.shape[-1] != 2:
        raise ValueError(
            "Hierarchical opinion reconstruction expects a final dimension of 2 "
            "for q_nonfactual and q_writer_given_nonfactual logits."
        )
    probabilities = torch.sigmoid(opinion_outputs)
    q_nonfactual = probabilities[..., 0]
    q_writer_given_nonfactual = probabilities[..., 1]
    return torch.stack(
        (
            1.0 - q_nonfactual,
            q_nonfactual * (1.0 - q_writer_given_nonfactual),
            q_nonfactual * q_writer_given_nonfactual,
        ),
        dim=-1,
    )


def opinion_probabilities_from_outputs(
    opinion_outputs: torch.Tensor,
    *,
    head_type: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    validate_opinion_head_type(head_type)
    if temperature <= 0.0:
        raise ValueError("Opinion temperature must be positive.")
    if head_type == LEGACY_FLAT_HEAD_TYPE:
        return torch.softmax(opinion_outputs / float(temperature), dim=-1)
    probabilities = reconstruct_hierarchical_opinion_probabilities(opinion_outputs)
    if temperature == 1.0:
        return probabilities
    calibration_logits = torch.log(probabilities.clamp_min(1e-12))
    return torch.softmax(calibration_logits / float(temperature), dim=-1)


def opinion_logits_for_calibration(
    opinion_outputs: torch.Tensor, *, head_type: str
) -> torch.Tensor:
    """Return three-class logits for the unchanged calibration policy."""

    validate_opinion_head_type(head_type)
    if head_type == LEGACY_FLAT_HEAD_TYPE:
        return opinion_outputs
    return torch.log(
        reconstruct_hierarchical_opinion_probabilities(opinion_outputs).clamp_min(1e-12)
    )


def build_classification_heads_checkpoint_payload(
    *,
    bias_head: torch.nn.Module,
    opinion_head: torch.nn.Module,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    bias_label2id: Mapping[str, int],
    opinion_label2id: Mapping[str, int],
    head_type: str,
) -> dict[str, Any]:
    """Build the explicit legacy v2 checkpoint payload."""

    architecture = opinion_head_architecture(
        head_type=head_type,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        opinion_label2id=opinion_label2id,
    )
    return {
        "checkpoint_schema_version": LEGACY_CLASSIFICATION_HEADS_SCHEMA_VERSION,
        "legacy_compatibility": True,
        "head_type": head_type,
        "bias_head_state_dict": bias_head.state_dict(),
        "opinion_head_state_dict": opinion_head.state_dict(),
        "embedding_dim": int(embedding_dim),
        "head_hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "bias_label2id": dict(bias_label2id),
        "opinion_label2id": dict(opinion_label2id),
        "opinion_head_architecture": architecture,
    }


def build_production_classification_heads_checkpoint_payload(
    *,
    bias_head: torch.nn.Module,
    opinion_style_head: torch.nn.Module,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
    bias_label2id: Mapping[str, int],
    opinion_style_label2id: Mapping[str, int],
) -> dict[str, Any]:
    expected_bias = {"Non-biased": 0, "Biased": 1}
    normalized_bias = {
        str(label): int(index) for label, index in bias_label2id.items()
    }
    normalized_style = {
        str(label): int(index) for label, index in opinion_style_label2id.items()
    }
    if normalized_bias != expected_bias:
        raise CheckpointCompatibilityError(
            f"Production bias mapping must be {expected_bias}, got {normalized_bias}."
        )
    if normalized_style != OPINION_STYLE_LABEL2ID:
        raise CheckpointCompatibilityError(
            "Production opinion-style mapping must be "
            f"{OPINION_STYLE_LABEL2ID}, got {normalized_style}."
        )
    bias_architecture = binary_head_architecture(
        head_type=BINARY_BIAS_HEAD_TYPE,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        logit_name="bias_logit",
    )
    style_architecture = binary_head_architecture(
        head_type=BINARY_OPINION_STYLE_HEAD_TYPE,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        logit_name="opinionated_style_logit",
    )
    return {
        "checkpoint_schema_version": CLASSIFICATION_HEADS_SCHEMA_VERSION,
        "classifier_input_contract": data_pipeline.CLASSIFIER_INPUT_CONTRACT,
        "classifier_mode": PRODUCTION_CLASSIFIER_MODE,
        "classifier_head_type": INDEPENDENT_BINARY_HEADS_TYPE,
        "legacy_compatible": False,
        "bias_head_type": BINARY_BIAS_HEAD_TYPE,
        "opinion_style_head_type": BINARY_OPINION_STYLE_HEAD_TYPE,
        "bias_head_state_dict": bias_head.state_dict(),
        "opinion_style_head_state_dict": opinion_style_head.state_dict(),
        "embedding_dim": int(embedding_dim),
        "head_hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "bias_label2id": normalized_bias,
        "opinion_style_label2id": normalized_style,
        "opinion_style_target_mapping": dict(
            data_pipeline.SOURCE_OPINION_TIER_TO_STYLE
        ),
        "bias_head_architecture": bias_architecture,
        "opinion_style_head_architecture": style_architecture,
    }


def build_linear_warmup_decay_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("Scheduler requires at least one optimizer step.")
    if warmup_ratio < 0 or warmup_ratio >= 1:
        raise ValueError("--warmup-ratio must be >= 0 and < 1.")

    warmup_steps = int(round(total_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        decay_steps = max(1, total_steps - warmup_steps)
        steps_after_warmup = current_step - warmup_steps
        return max(0.0, 1.0 - (steps_after_warmup / decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_class_weights(
    labels: pd.Series, num_labels: int, device: torch.device
) -> torch.Tensor:
    observed_labels = labels.dropna().astype(int)
    counts = np.bincount(observed_labels.to_numpy(), minlength=num_labels)
    if (counts == 0).any():
        raise ValueError("Cannot build class weights because a class has zero train rows.")

    total = int(counts.sum())
    weights = total / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loss_functions(
    train_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    bias_num_labels: int,
    opinion_num_labels: int,
) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    torch.Tensor | None,
    dict[str, Any],
]:
    bias_weights = None
    opinion_weights = None
    if args.class_weighting == "balanced":
        bias_weights = build_class_weights(
            train_df["bias_label_id"], bias_num_labels, device
        )
        if not (
            args.opinion_head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE
            and args.opinion_sampling == "balanced"
        ):
            opinion_weights = build_class_weights(
                train_df["opinion_label_id"], opinion_num_labels, device
            )

    weight_summary = {
        "bias": bias_weights.detach().cpu().tolist() if bias_weights is not None else None,
        "opinion": (
            opinion_weights.detach().cpu().tolist()
            if opinion_weights is not None
            else None
        ),
        "opinion_weighting_source": (
            "sampler"
            if (
                args.opinion_head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE
                and args.opinion_sampling == "balanced"
            )
            else args.class_weighting
        ),
        "middle_class_weight": float(args.middle_class_weight),
        "opinion_focal_gamma": float(args.opinion_focal_gamma),
    }
    opinion_criterion: torch.nn.Module
    if args.opinion_head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE:
        opinion_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    else:
        opinion_criterion = torch.nn.CrossEntropyLoss(
            weight=opinion_weights,
            ignore_index=LABEL_IGNORE_INDEX,
            reduction="none",
        )
    return (
        torch.nn.CrossEntropyLoss(
            weight=bias_weights,
            ignore_index=LABEL_IGNORE_INDEX,
            reduction="none",
        ),
        opinion_criterion,
        opinion_weights,
        weight_summary,
    )


def build_production_loss_functions(
    train_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    bias_weights = None
    style_weights = None
    if args.class_weighting == "balanced":
        bias_weights = build_class_weights(train_df["bias_label_id"], 2, device)
        style_weights = build_class_weights(
            train_df["opinion_style_label_id"], 2, device
        )
    summary = {
        "bias": bias_weights.detach().cpu().tolist() if bias_weights is not None else None,
        "opinion_style": (
            style_weights.detach().cpu().tolist() if style_weights is not None else None
        ),
        "weighting_source": args.class_weighting,
    }
    return (
        torch.nn.BCEWithLogitsLoss(reduction="none"),
        torch.nn.BCEWithLogitsLoss(reduction="none"),
        bias_weights,
        style_weights,
        summary,
    )


def build_opinion_balanced_sampler(
    train_df: pd.DataFrame, args: argparse.Namespace
) -> Any | None:
    if args.opinion_sampling == "none":
        return None
    if args.opinion_sampling != "balanced":
        raise ValueError(f"Unsupported opinion sampling mode: {args.opinion_sampling!r}.")
    if args.opinion_head_type != HIERARCHICAL_ORDINAL_HEAD_TYPE:
        raise ValueError(
            "Balanced opinion sampling requires the hierarchical opinion-head mode."
        )
    if WeightedRandomSampler is None:
        raise RuntimeError("Training dependencies were not initialized for sampling.")
    labels = train_df["opinion_label_id"]
    valid_labels = labels.dropna().astype(int)
    counts = np.bincount(
        valid_labels.to_numpy(), minlength=len(CANONICAL_OPINION_LABELS)
    )
    if len(counts) != len(CANONICAL_OPINION_LABELS) or (counts == 0).any():
        raise ValueError(
            "Balanced opinion sampling requires at least one train row for every "
            "canonical opinion tier."
        )
    total_labeled = int(counts.sum())
    inverse_frequency = total_labeled / (len(counts) * counts)
    weights: list[float] = []
    for value in labels.tolist():
        if pd.isna(value):
            weights.append(1.0)
        else:
            weights.append(float(inverse_frequency[int(value)]))
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(train_df),
        replacement=True,
        generator=generator,
    )


def classification_metrics(
    predictions: list[int], labels: list[int], num_labels: int
) -> dict[str, object]:
    if not labels:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": [],
            "confusion_matrix": [],
        }
    return foundation_evaluation.classification_metrics(
        labels,
        predictions,
        labels=list(range(num_labels)),
    )


def _validate_probability_threshold(name: str, value: float) -> float:
    threshold = float(value)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {threshold}.")
    return threshold


def _binary_roc_auc(
    truth: Sequence[int], scores: Sequence[float]
) -> float | None:
    """Return tie-aware ROC-AUC, or None when one class is absent."""

    if len(truth) != len(scores):
        raise ValueError("Binary truth and scores must have the same length.")
    positive_count = sum(int(value) == 1 for value in truth)
    negative_count = len(truth) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    ranked = sorted(
        ((float(score), int(label)) for score, label in zip(scores, truth)),
        key=lambda item: item[0],
    )
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label == 1 for _score, label in ranked[index:end]
        )
        index = end
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _binary_pr_auc(truth: Sequence[int], scores: Sequence[float]) -> float | None:
    """Return average precision (step-wise PR-AUC), or None for one class."""

    if len(truth) != len(scores):
        raise ValueError("Binary truth and scores must have the same length.")
    positive_count = sum(int(value) == 1 for value in truth)
    negative_count = len(truth) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    ranked = sorted(
        ((float(score), int(label)) for score, label in zip(scores, truth)),
        key=lambda item: item[0],
        reverse=True,
    )
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        for _score, label in ranked[index:end]:
            if label == 1:
                true_positive += 1
            else:
                false_positive += 1
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / positive_count
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


def _classification_diagnostics(
    truth: Sequence[int],
    predictions: Sequence[int],
    *,
    labels: Sequence[int],
    label_names: Mapping[int, str],
) -> dict[str, object]:
    metrics = foundation_evaluation.classification_metrics(
        truth,
        predictions,
        labels=labels,
        label_names=label_names,
    )
    per_class = list(metrics["per_class"])
    observed_recalls = [
        float(item["recall"]) for item in per_class if int(item["support"]) > 0
    ]
    return {
        "support": int(len(truth)),
        "true_class_counts": {
            label_names[int(label)]: int(sum(int(value) == int(label) for value in truth))
            for label in labels
        },
        "predicted_class_counts": {
            label_names[int(label)]: int(
                sum(int(value) == int(label) for value in predictions)
            )
            for label in labels
        },
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": (
            sum(observed_recalls) / len(observed_recalls) if observed_recalls else 0.0
        ),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class": per_class,
        "confusion_matrix": metrics["confusion_matrix"],
    }


def hierarchical_opinion_evaluation_diagnostics(
    opinion_outputs: torch.Tensor,
    opinion_labels: torch.Tensor,
    *,
    stage_thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Evaluate hierarchical stages independently and as routed three-class outputs.

    Stage two is oracle-gated by the true middle/writer label. Thresholds are
    reporting-only; callers may supply validation/calibration thresholds, and
    otherwise both stages use the fixed 0.5 default.
    """

    if opinion_outputs.ndim != 2 or opinion_outputs.shape[-1] != 2:
        raise ValueError("Hierarchical diagnostics expect [batch, 2] logits.")
    if opinion_labels.ndim != 1 or opinion_labels.shape[0] != opinion_outputs.shape[0]:
        raise ValueError("Hierarchical diagnostics require one label per logit row.")

    thresholds = dict(stage_thresholds or {})
    nonfactual_threshold = _validate_probability_threshold(
        "q_nonfactual threshold", thresholds.get("q_nonfactual", 0.5)
    )
    writer_threshold = _validate_probability_threshold(
        "q_writer_given_nonfactual threshold",
        thresholds.get("q_writer_given_nonfactual", 0.5),
    )
    factual_id = CANONICAL_OPINION_LABEL2ID[CANONICAL_OPINION_LABELS[0]]
    middle_id = CANONICAL_OPINION_LABEL2ID[CANONICAL_OPINION_LABELS[1]]
    writer_id = CANONICAL_OPINION_LABEL2ID[CANONICAL_OPINION_LABELS[2]]
    valid_mask = opinion_labels.ne(LABEL_IGNORE_INDEX)
    valid_labels = opinion_labels[valid_mask].to(dtype=torch.long)
    if bool(valid_mask.any().item()):
        if bool(((valid_labels < factual_id) | (valid_labels > writer_id)).any().item()):
            raise ValueError("Hierarchical diagnostics require opinion labels 0, 1, or 2.")

    probabilities = torch.sigmoid(opinion_outputs[valid_mask]).detach().cpu()
    labels = valid_labels.detach().cpu()
    q_nonfactual = probabilities[:, 0]
    q_writer_given_nonfactual = probabilities[:, 1]
    nonfactual_truth = labels.ne(factual_id).to(dtype=torch.long)
    nonfactual_predictions = q_nonfactual.ge(nonfactual_threshold).to(dtype=torch.long)
    nonfactual_truth_list = [int(value) for value in nonfactual_truth.tolist()]
    nonfactual_prediction_list = [int(value) for value in nonfactual_predictions.tolist()]
    nonfactual_score_list = [float(value) for value in q_nonfactual.tolist()]

    stage_two_mask = labels.ne(factual_id)
    stage_two_truth = labels[stage_two_mask].eq(writer_id).to(dtype=torch.long)
    stage_two_predictions = q_writer_given_nonfactual[stage_two_mask].ge(
        writer_threshold
    ).to(dtype=torch.long)
    stage_two_truth_list = [int(value) for value in stage_two_truth.tolist()]
    stage_two_prediction_list = [int(value) for value in stage_two_predictions.tolist()]
    stage_two_score_list = [
        float(value) for value in q_writer_given_nonfactual[stage_two_mask].tolist()
    ]

    routed_predictions = torch.full_like(labels, factual_id)
    routed_nonfactual = q_nonfactual.ge(nonfactual_threshold)
    routed_predictions[routed_nonfactual] = middle_id
    routed_predictions[
        routed_nonfactual & q_writer_given_nonfactual.ge(writer_threshold)
    ] = writer_id
    reconstructed_probabilities = torch.stack(
        (
            1.0 - q_nonfactual,
            q_nonfactual * (1.0 - q_writer_given_nonfactual),
            q_nonfactual * q_writer_given_nonfactual,
        ),
        dim=-1,
    )
    probability_predictions = reconstructed_probabilities.argmax(dim=-1)
    three_class_names = {
        factual_id: CANONICAL_OPINION_LABELS[0],
        middle_id: CANONICAL_OPINION_LABELS[1],
        writer_id: CANONICAL_OPINION_LABELS[2],
    }
    label_list = [int(value) for value in labels.tolist()]

    nonfactual = _classification_diagnostics(
        nonfactual_truth_list,
        nonfactual_prediction_list,
        labels=(0, 1),
        label_names={0: "factual", 1: "non_factual"},
    )
    nonfactual.update(
        {
            "positive_label": "non_factual",
            "threshold": nonfactual_threshold,
            "roc_auc": _binary_roc_auc(nonfactual_truth_list, nonfactual_score_list),
            "pr_auc": _binary_pr_auc(nonfactual_truth_list, nonfactual_score_list),
            "pr_auc_definition": "average_precision",
        }
    )
    writer_given_nonfactual = _classification_diagnostics(
        stage_two_truth_list,
        stage_two_prediction_list,
        labels=(0, 1),
        label_names={0: "middle", 1: "writer"},
    )
    writer_given_nonfactual.update(
        {
            "positive_label": "writer",
            "threshold": writer_threshold,
            "oracle_gated_by_true_nonfactual": True,
            "roc_auc": _binary_roc_auc(stage_two_truth_list, stage_two_score_list),
            "pr_auc": _binary_pr_auc(stage_two_truth_list, stage_two_score_list),
            "pr_auc_definition": "average_precision",
        }
    )
    routed = _classification_diagnostics(
        label_list,
        [int(value) for value in routed_predictions.tolist()],
        labels=(factual_id, middle_id, writer_id),
        label_names=three_class_names,
    )
    routed.update(
        {
            "q_nonfactual_threshold": nonfactual_threshold,
            "q_writer_given_nonfactual_threshold": writer_threshold,
            "prediction_rule": "threshold_routed/v1",
        }
    )
    final_probabilities = _classification_diagnostics(
        label_list,
        [int(value) for value in probability_predictions.tolist()],
        labels=(factual_id, middle_id, writer_id),
        label_names=three_class_names,
    )
    final_probabilities.update(
        {
            "prediction_rule": "argmax(reconstructed_probabilities)",
            "probability_reconstruction": {
                "factual": "1 - q_nonfactual",
                "middle": "q_nonfactual * (1 - q_writer_given_nonfactual)",
                "writer": "q_nonfactual * q_writer_given_nonfactual",
            },
        }
    )
    return {
        "schema_version": "hierarchical_opinion_evaluation/v1",
        "threshold_source": (
            "validation_or_calibration" if stage_thresholds else "default_0.5"
        ),
        "q_nonfactual": nonfactual,
        "q_writer_given_nonfactual": writer_given_nonfactual,
        "threshold_routed_pipeline": routed,
        "final_three_class_probabilities": final_probabilities,
    }


def log_hierarchical_opinion_diagnostics(
    evaluation_name: str, diagnostics: Mapping[str, object]
) -> None:
    style = diagnostics["q_nonfactual"]
    middle_writer = diagnostics["q_writer_given_nonfactual"]
    routed = diagnostics["threshold_routed_pipeline"]
    LOGGER.info(
        "%s legacy hierarchy style_macro_f1=%.4f middle_writer_macro_f1=%.4f "
        "three_tier_macro_f1=%.4f",
        evaluation_name,
        float(style["macro_f1"]),
        float(middle_writer["macro_f1"]),
        float(routed["macro_f1"]),
    )


def sentence_embeddings(
    model: SentenceTransformer, texts: list[str], device: torch.device
) -> torch.Tensor:
    features = model.tokenize(texts)
    features = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in features.items()
    }
    return model(features)["sentence_embedding"]


def _masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not bool(mask.any().item()):
        return values.sum() * 0.0
    selected = values[mask]
    if sample_weights is None:
        return selected.mean()
    selected_weights = sample_weights[mask]
    return (selected * selected_weights).sum() / selected_weights.sum().clamp_min(1e-12)


def _binary_sample_weights(
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor | None:
    if class_weights is None:
        return None
    weights = torch.ones_like(labels, dtype=torch.float32)
    if bool(mask.any().item()):
        weights[mask] = class_weights[labels[mask]]
    return weights


def production_batch_loss(
    model: SentenceTransformer,
    bias_head: torch.nn.Module,
    opinion_style_head: torch.nn.Module,
    batch: dict[str, object],
    bias_criterion: torch.nn.Module,
    opinion_style_criterion: torch.nn.Module,
    bias_class_weights: torch.Tensor | None,
    opinion_style_class_weights: torch.Tensor | None,
    opinion_style_loss_weight: float,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, int],
]:
    """Run one encoder pass and two independent scalar production heads."""

    embeddings = sentence_embeddings(model, batch["texts"], device)
    bias_logits = bias_head(embeddings).squeeze(-1)
    opinion_style_logits = opinion_style_head(embeddings).squeeze(-1)
    if bias_logits.ndim != 1 or opinion_style_logits.ndim != 1:
        raise ValueError("Production binary heads must emit one scalar logit per row.")

    bias_labels = batch["bias_labels"].to(device)
    style_labels = batch["opinion_style_labels"].to(device)
    bias_mask = bias_labels.ne(LABEL_IGNORE_INDEX)
    style_mask = style_labels.ne(LABEL_IGNORE_INDEX)
    bias_values = bias_criterion(
        bias_logits, bias_labels.clamp_min(0).to(dtype=bias_logits.dtype)
    )
    style_values = opinion_style_criterion(
        opinion_style_logits,
        style_labels.clamp_min(0).to(dtype=opinion_style_logits.dtype),
    )
    bias_loss = _masked_weighted_mean(
        bias_values,
        bias_mask,
        _binary_sample_weights(bias_labels, bias_mask, bias_class_weights),
    )
    opinion_style_loss = _masked_weighted_mean(
        style_values,
        style_mask,
        _binary_sample_weights(
            style_labels, style_mask, opinion_style_class_weights
        ),
    )
    total_loss = bias_loss + float(opinion_style_loss_weight) * opinion_style_loss
    return (
        total_loss,
        bias_loss,
        opinion_style_loss,
        bias_logits,
        opinion_style_logits,
        {
            "bias_supervised_examples": int(bias_mask.sum().item()),
            "opinion_style_supervised_examples": int(style_mask.sum().item()),
        },
    )


def hierarchical_opinion_loss(
    opinion_outputs: torch.Tensor,
    opinion_labels: torch.Tensor,
    *,
    middle_class_weight: float,
    focal_gamma: float,
    opinion_class_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    """Compute the two BCE terms for canonical factual/middle/writer labels."""

    if opinion_outputs.ndim != 2 or opinion_outputs.shape[-1] != 2:
        raise ValueError("Hierarchical opinion loss expects [batch, 2] logits.")
    if middle_class_weight <= 0.0:
        raise ValueError("middle_class_weight must be positive.")
    if focal_gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative.")

    supervised_mask = opinion_labels.ne(LABEL_IGNORE_INDEX)
    labels = opinion_labels.clamp_min(0)
    factual_id = CANONICAL_OPINION_LABEL2ID[CANONICAL_OPINION_LABELS[0]]
    middle_id = CANONICAL_OPINION_LABEL2ID[CANONICAL_OPINION_LABELS[1]]
    writer_id = CANONICAL_OPINION_LABEL2ID[CANONICAL_OPINION_LABELS[2]]
    if bool(supervised_mask.any().item()):
        observed = labels[supervised_mask]
        if bool(((observed < factual_id) | (observed > writer_id)).any().item()):
            raise ValueError("Hierarchical opinion labels must be 0=factual, 1=middle, 2=writer.")

    sample_weights = torch.ones_like(opinion_outputs[:, 0], dtype=torch.float32)
    if opinion_class_weights is not None and bool(supervised_mask.any().item()):
        sample_weights[supervised_mask] = opinion_class_weights[labels[supervised_mask]]
    middle_mask = supervised_mask & labels.eq(middle_id)
    sample_weights[middle_mask] *= float(middle_class_weight)

    nonfactual_target = labels.ne(factual_id).to(dtype=opinion_outputs.dtype)
    nonfactual_values = torch.nn.functional.binary_cross_entropy_with_logits(
        opinion_outputs[:, 0], nonfactual_target, reduction="none"
    )
    conditional_mask = supervised_mask & labels.ne(factual_id)
    writer_target = labels.eq(writer_id).to(dtype=opinion_outputs.dtype)
    conditional_values = torch.nn.functional.binary_cross_entropy_with_logits(
        opinion_outputs[:, 1], writer_target, reduction="none"
    )
    if focal_gamma > 0.0:
        nonfactual_probabilities = torch.sigmoid(opinion_outputs[:, 0])
        nonfactual_pt = torch.where(
            nonfactual_target.bool(), nonfactual_probabilities, 1.0 - nonfactual_probabilities
        )
        nonfactual_values = nonfactual_values * (1.0 - nonfactual_pt).pow(focal_gamma)
        conditional_probabilities = torch.sigmoid(opinion_outputs[:, 1])
        conditional_pt = torch.where(
            writer_target.bool(), conditional_probabilities, 1.0 - conditional_probabilities
        )
        conditional_values = conditional_values * (1.0 - conditional_pt).pow(focal_gamma)

    nonfactual_loss = _masked_weighted_mean(
        nonfactual_values, supervised_mask, sample_weights
    )
    conditional_loss = _masked_weighted_mean(
        conditional_values, conditional_mask, sample_weights
    )
    return (
        nonfactual_loss + conditional_loss,
        nonfactual_loss,
        conditional_loss,
        {
            "opinion_supervised_examples": int(supervised_mask.sum().item()),
            "conditional_opinion_examples": int(conditional_mask.sum().item()),
        },
    )


def batch_loss(
    model: SentenceTransformer,
    bias_head: torch.nn.Module,
    opinion_head: torch.nn.Module,
    batch: dict[str, object],
    bias_criterion: torch.nn.Module,
    opinion_criterion: torch.nn.Module,
    opinion_class_weights: torch.Tensor | None,
    opinion_loss_alpha: float,
    opinion_head_type: str,
    middle_class_weight: float,
    opinion_focal_gamma: float,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, int | torch.Tensor],
]:
    embeddings = sentence_embeddings(model, batch["texts"], device)
    bias_logits = bias_head(embeddings)
    opinion_outputs = opinion_head(embeddings)
    bias_labels = batch["bias_labels"].to(device)
    opinion_labels = batch["opinion_labels"].to(device)

    bias_mask = bias_labels.ne(LABEL_IGNORE_INDEX)
    bias_loss = _masked_weighted_mean(bias_criterion(bias_logits, bias_labels), bias_mask)
    if opinion_head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE:
        (
            opinion_loss,
            nonfactual_loss,
            conditional_loss,
            opinion_counts,
        ) = hierarchical_opinion_loss(
            opinion_outputs,
            opinion_labels,
            middle_class_weight=middle_class_weight,
            focal_gamma=opinion_focal_gamma,
            opinion_class_weights=opinion_class_weights,
        )
    else:
        opinion_mask = opinion_labels.ne(LABEL_IGNORE_INDEX)
        opinion_loss = _masked_weighted_mean(
            opinion_criterion(opinion_outputs, opinion_labels), opinion_mask
        )
        nonfactual_loss = opinion_loss
        conditional_loss = opinion_loss.sum() * 0.0
        opinion_counts = {
            "opinion_supervised_examples": int(opinion_mask.sum().item()),
            "conditional_opinion_examples": 0,
        }
    total_loss = bias_loss + opinion_loss_alpha * opinion_loss
    component_counts: dict[str, int | torch.Tensor] = {
        "bias_supervised_examples": int(bias_mask.sum().item()),
        "opinion_nonfactual_loss": nonfactual_loss,
        "opinion_conditional_loss": conditional_loss,
        **opinion_counts,
    }
    return (
        total_loss,
        bias_loss,
        opinion_loss,
        bias_logits,
        opinion_outputs,
        component_counts,
    )


def train_one_epoch(
    model: SentenceTransformer,
    bias_head: torch.nn.Module,
    opinion_head: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    bias_criterion: torch.nn.Module,
    opinion_criterion: torch.nn.Module,
    opinion_class_weights: torch.Tensor | None,
    opinion_loss_alpha: float,
    opinion_head_type: str,
    middle_class_weight: float,
    opinion_focal_gamma: float,
    device: torch.device,
    use_amp: bool,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    bias_head.train()
    opinion_head.train()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    totals = {
        "loss": 0.0,
        "bias_loss": 0.0,
        "opinion_loss": 0.0,
        "opinion_nonfactual_loss": 0.0,
        "opinion_conditional_loss": 0.0,
        "bias_supervised_examples": 0,
        "opinion_supervised_examples": 0,
        "conditional_opinion_examples": 0,
        "examples": 0,
    }

    for batch in dataloader:
        batch_size = len(batch["texts"])
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            (
                total_loss,
                bias_loss,
                opinion_loss,
                _,
                _,
                loss_components,
            ) = batch_loss(
                model=model,
                bias_head=bias_head,
                opinion_head=opinion_head,
                batch=batch,
                bias_criterion=bias_criterion,
                opinion_criterion=opinion_criterion,
                opinion_class_weights=opinion_class_weights,
                opinion_loss_alpha=opinion_loss_alpha,
                opinion_head_type=opinion_head_type,
                middle_class_weight=middle_class_weight,
                opinion_focal_gamma=opinion_focal_gamma,
                device=device,
            )

        scaler.scale(total_loss).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(bias_head.parameters())
                + list(opinion_head.parameters()),
                max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        totals["loss"] += float(total_loss.detach().cpu()) * batch_size
        totals["bias_loss"] += float(bias_loss.detach().cpu()) * batch_size
        totals["opinion_loss"] += float(opinion_loss.detach().cpu()) * batch_size
        totals["opinion_nonfactual_loss"] += (
            float(loss_components["opinion_nonfactual_loss"].detach().cpu()) * batch_size
        )
        totals["opinion_conditional_loss"] += (
            float(loss_components["opinion_conditional_loss"].detach().cpu()) * batch_size
        )
        totals["bias_supervised_examples"] += int(loss_components["bias_supervised_examples"])
        totals["opinion_supervised_examples"] += int(
            loss_components["opinion_supervised_examples"]
        )
        totals["conditional_opinion_examples"] += int(
            loss_components["conditional_opinion_examples"]
        )
        totals["examples"] += batch_size

    examples = max(1, totals["examples"])
    return {
        "loss": totals["loss"] / examples,
        "bias_loss": totals["bias_loss"] / examples,
        "opinion_loss": totals["opinion_loss"] / examples,
        "opinion_nonfactual_loss": totals["opinion_nonfactual_loss"] / examples,
        "opinion_conditional_loss": totals["opinion_conditional_loss"] / examples,
        "bias_supervised_examples": float(totals["bias_supervised_examples"]),
        "opinion_supervised_examples": float(totals["opinion_supervised_examples"]),
        "conditional_opinion_examples": float(totals["conditional_opinion_examples"]),
        "learning_rate": optimizer.param_groups[0]["lr"],
    }


def _per_class_metric(
    metrics: Mapping[str, object], label_id: int, metric_name: str
) -> float:
    for item in metrics.get("per_class", []):
        if int(item["label_id"]) == int(label_id):
            return float(item[metric_name])
    return 0.0


def _production_metrics(
    *,
    totals: Mapping[str, float | int],
    bias_truth: Sequence[int],
    bias_probabilities: Sequence[float],
    opinion_style_truth: Sequence[int],
    opinion_style_probabilities: Sequence[float],
    include_learning_rate: float | None = None,
) -> dict[str, object]:
    examples = max(1, int(totals["examples"]))
    bias_metrics = foundation_evaluation.binary_classification_metrics(
        bias_truth,
        bias_probabilities,
        negative_label="not_biased",
        positive_label="biased",
    )
    style_metrics = foundation_evaluation.binary_classification_metrics(
        opinion_style_truth,
        opinion_style_probabilities,
        negative_label="objective_style",
        positive_label="opinionated_style",
    )
    result: dict[str, object] = {
        "loss": float(totals["loss"]) / examples,
        "bias_loss": float(totals["bias_loss"]) / examples,
        "opinion_style_loss": float(totals["opinion_style_loss"]) / examples,
        "bias_supervised_examples": int(totals["bias_supervised_examples"]),
        "opinion_style_supervised_examples": int(
            totals["opinion_style_supervised_examples"]
        ),
        "bias": bias_metrics,
        "opinion_style": style_metrics,
        "bias_accuracy": float(bias_metrics["accuracy"]),
        "bias_balanced_accuracy": float(bias_metrics["balanced_accuracy"]),
        "bias_macro_f1": float(bias_metrics["macro_f1"]),
        "bias_biased_precision": _per_class_metric(bias_metrics, 1, "precision"),
        "bias_biased_recall": _per_class_metric(bias_metrics, 1, "recall"),
        "opinion_style_accuracy": float(style_metrics["accuracy"]),
        "opinion_style_balanced_accuracy": float(
            style_metrics["balanced_accuracy"]
        ),
        "opinion_style_macro_f1": float(style_metrics["macro_f1"]),
        "opinion_style_objective_recall": _per_class_metric(
            style_metrics, 0, "recall"
        ),
        "opinion_style_opinionated_recall": _per_class_metric(
            style_metrics, 1, "recall"
        ),
    }
    if include_learning_rate is not None:
        result["learning_rate"] = float(include_learning_rate)
    return result


def train_production_one_epoch(
    model: SentenceTransformer,
    bias_head: torch.nn.Module,
    opinion_style_head: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    bias_criterion: torch.nn.Module,
    opinion_style_criterion: torch.nn.Module,
    bias_class_weights: torch.Tensor | None,
    opinion_style_class_weights: torch.Tensor | None,
    opinion_style_loss_weight: float,
    device: torch.device,
    use_amp: bool,
    max_grad_norm: float,
) -> dict[str, object]:
    model.train()
    bias_head.train()
    opinion_style_head.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    totals: dict[str, float | int] = {
        "loss": 0.0,
        "bias_loss": 0.0,
        "opinion_style_loss": 0.0,
        "bias_supervised_examples": 0,
        "opinion_style_supervised_examples": 0,
        "examples": 0,
    }
    bias_truth: list[int] = []
    bias_probabilities: list[float] = []
    style_truth: list[int] = []
    style_probabilities: list[float] = []
    for batch in dataloader:
        batch_size = len(batch["texts"])
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            (
                total_loss,
                bias_loss,
                opinion_style_loss,
                bias_logits,
                opinion_style_logits,
                counts,
            ) = production_batch_loss(
                model,
                bias_head,
                opinion_style_head,
                batch,
                bias_criterion,
                opinion_style_criterion,
                bias_class_weights,
                opinion_style_class_weights,
                opinion_style_loss_weight,
                device,
            )
        scaler.scale(total_loss).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(bias_head.parameters())
                + list(opinion_style_head.parameters()),
                max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        totals["loss"] += float(total_loss.detach().cpu()) * batch_size
        totals["bias_loss"] += float(bias_loss.detach().cpu()) * batch_size
        totals["opinion_style_loss"] += (
            float(opinion_style_loss.detach().cpu()) * batch_size
        )
        totals["bias_supervised_examples"] += counts["bias_supervised_examples"]
        totals["opinion_style_supervised_examples"] += counts[
            "opinion_style_supervised_examples"
        ]
        totals["examples"] += batch_size
        bias_labels = batch["bias_labels"]
        style_labels = batch["opinion_style_labels"]
        bias_mask = bias_labels.ne(LABEL_IGNORE_INDEX)
        style_mask = style_labels.ne(LABEL_IGNORE_INDEX)
        bias_truth.extend(bias_labels[bias_mask].tolist())
        style_truth.extend(style_labels[style_mask].tolist())
        bias_probabilities.extend(
            torch.sigmoid(bias_logits.detach().cpu())[bias_mask].tolist()
        )
        style_probabilities.extend(
            torch.sigmoid(opinion_style_logits.detach().cpu())[style_mask].tolist()
        )
    return _production_metrics(
        totals=totals,
        bias_truth=bias_truth,
        bias_probabilities=bias_probabilities,
        opinion_style_truth=style_truth,
        opinion_style_probabilities=style_probabilities,
        include_learning_rate=optimizer.param_groups[0]["lr"],
    )


def evaluate_production(
    model: SentenceTransformer,
    bias_head: torch.nn.Module,
    opinion_style_head: torch.nn.Module,
    dataloader: DataLoader,
    bias_criterion: torch.nn.Module,
    opinion_style_criterion: torch.nn.Module,
    bias_class_weights: torch.Tensor | None,
    opinion_style_class_weights: torch.Tensor | None,
    opinion_style_loss_weight: float,
    device: torch.device,
    use_amp: bool,
) -> dict[str, object]:
    if len(dataloader) == 0:
        return {}
    model.eval()
    bias_head.eval()
    opinion_style_head.eval()
    totals: dict[str, float | int] = {
        "loss": 0.0,
        "bias_loss": 0.0,
        "opinion_style_loss": 0.0,
        "bias_supervised_examples": 0,
        "opinion_style_supervised_examples": 0,
        "examples": 0,
    }
    bias_truth: list[int] = []
    bias_probabilities: list[float] = []
    style_truth: list[int] = []
    style_probabilities: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch_size = len(batch["texts"])
            with torch.cuda.amp.autocast(enabled=use_amp):
                (
                    total_loss,
                    bias_loss,
                    opinion_style_loss,
                    bias_logits,
                    opinion_style_logits,
                    counts,
                ) = production_batch_loss(
                    model,
                    bias_head,
                    opinion_style_head,
                    batch,
                    bias_criterion,
                    opinion_style_criterion,
                    bias_class_weights,
                    opinion_style_class_weights,
                    opinion_style_loss_weight,
                    device,
                )
            totals["loss"] += float(total_loss.cpu()) * batch_size
            totals["bias_loss"] += float(bias_loss.cpu()) * batch_size
            totals["opinion_style_loss"] += (
                float(opinion_style_loss.cpu()) * batch_size
            )
            totals["bias_supervised_examples"] += counts[
                "bias_supervised_examples"
            ]
            totals["opinion_style_supervised_examples"] += counts[
                "opinion_style_supervised_examples"
            ]
            totals["examples"] += batch_size
            bias_labels = batch["bias_labels"]
            style_labels = batch["opinion_style_labels"]
            bias_mask = bias_labels.ne(LABEL_IGNORE_INDEX)
            style_mask = style_labels.ne(LABEL_IGNORE_INDEX)
            bias_truth.extend(bias_labels[bias_mask].tolist())
            style_truth.extend(style_labels[style_mask].tolist())
            bias_probabilities.extend(
                torch.sigmoid(bias_logits.detach().cpu())[bias_mask].tolist()
            )
            style_probabilities.extend(
                torch.sigmoid(opinion_style_logits.detach().cpu())[style_mask].tolist()
            )
    return _production_metrics(
        totals=totals,
        bias_truth=bias_truth,
        bias_probabilities=bias_probabilities,
        opinion_style_truth=style_truth,
        opinion_style_probabilities=style_probabilities,
    )


def evaluate(
    model: SentenceTransformer,
    bias_head: torch.nn.Module,
    opinion_head: torch.nn.Module,
    dataloader: DataLoader,
    bias_criterion: torch.nn.Module,
    opinion_criterion: torch.nn.Module,
    opinion_class_weights: torch.Tensor | None,
    opinion_loss_alpha: float,
    opinion_head_type: str,
    middle_class_weight: float,
    opinion_focal_gamma: float,
    device: torch.device,
    use_amp: bool,
    bias_num_labels: int,
    opinion_num_labels: int,
    hierarchical_stage_thresholds: Mapping[str, float] | None = None,
    evaluation_name: str = "evaluation",
) -> dict[str, object]:
    if len(dataloader) == 0:
        return {}

    model.eval()
    bias_head.eval()
    opinion_head.eval()

    totals = {
        "loss": 0.0,
        "bias_loss": 0.0,
        "opinion_loss": 0.0,
        "opinion_nonfactual_loss": 0.0,
        "opinion_conditional_loss": 0.0,
        "bias_correct": 0,
        "opinion_correct": 0,
        "bias_supervised_examples": 0,
        "opinion_supervised_examples": 0,
        "conditional_opinion_examples": 0,
        "examples": 0,
    }
    bias_predictions: list[int] = []
    bias_targets: list[int] = []
    opinion_predictions: list[int] = []
    opinion_targets: list[int] = []
    hierarchical_opinion_outputs: list[torch.Tensor] = []
    hierarchical_opinion_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            batch_size = len(batch["texts"])
            with torch.cuda.amp.autocast(enabled=use_amp):
                (
                    total_loss,
                    bias_loss,
                    opinion_loss,
                    bias_logits,
                    opinion_outputs,
                    loss_components,
                ) = batch_loss(
                    model=model,
                    bias_head=bias_head,
                    opinion_head=opinion_head,
                    batch=batch,
                    bias_criterion=bias_criterion,
                    opinion_criterion=opinion_criterion,
                    opinion_class_weights=opinion_class_weights,
                    opinion_loss_alpha=opinion_loss_alpha,
                    opinion_head_type=opinion_head_type,
                    middle_class_weight=middle_class_weight,
                    opinion_focal_gamma=opinion_focal_gamma,
                    device=device,
                )

            bias_labels = batch["bias_labels"].to(device)
            opinion_labels = batch["opinion_labels"].to(device)
            if opinion_head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE:
                hierarchical_opinion_outputs.append(opinion_outputs.detach().cpu())
                hierarchical_opinion_labels.append(opinion_labels.detach().cpu())
            bias_mask = bias_labels.ne(LABEL_IGNORE_INDEX)
            opinion_mask = opinion_labels.ne(LABEL_IGNORE_INDEX)
            batch_bias_predictions = bias_logits.argmax(dim=-1)
            batch_opinion_predictions = opinion_probabilities_from_outputs(
                opinion_outputs, head_type=opinion_head_type
            ).argmax(dim=-1)
            totals["loss"] += float(total_loss.cpu()) * batch_size
            totals["bias_loss"] += float(bias_loss.cpu()) * batch_size
            totals["opinion_loss"] += float(opinion_loss.cpu()) * batch_size
            totals["opinion_nonfactual_loss"] += (
                float(loss_components["opinion_nonfactual_loss"].cpu()) * batch_size
            )
            totals["opinion_conditional_loss"] += (
                float(loss_components["opinion_conditional_loss"].cpu()) * batch_size
            )
            totals["bias_correct"] += int(
                (batch_bias_predictions[bias_mask] == bias_labels[bias_mask]).sum().cpu()
            )
            totals["opinion_correct"] += int(
                (
                    batch_opinion_predictions[opinion_mask]
                    == opinion_labels[opinion_mask]
                ).sum().cpu()
            )
            totals["bias_supervised_examples"] += int(bias_mask.sum().cpu())
            totals["opinion_supervised_examples"] += int(opinion_mask.sum().cpu())
            totals["conditional_opinion_examples"] += int(
                loss_components["conditional_opinion_examples"]
            )
            totals["examples"] += batch_size
            bias_predictions.extend(batch_bias_predictions[bias_mask].cpu().tolist())
            bias_targets.extend(bias_labels[bias_mask].cpu().tolist())
            opinion_predictions.extend(
                batch_opinion_predictions[opinion_mask].cpu().tolist()
            )
            opinion_targets.extend(opinion_labels[opinion_mask].cpu().tolist())

    examples = max(1, totals["examples"])
    bias_metrics = classification_metrics(
        bias_predictions, bias_targets, bias_num_labels
    )
    opinion_metrics = classification_metrics(
        opinion_predictions, opinion_targets, opinion_num_labels
    )
    middle_recall = 0.0
    for metrics in opinion_metrics["per_class"]:
        if metrics.get("label_id") == CANONICAL_OPINION_LABEL2ID[CANONICAL_OPINION_LABELS[1]]:
            middle_recall = float(metrics["recall"])
            break
    bias_examples = max(1, int(totals["bias_supervised_examples"]))
    opinion_examples = max(1, int(totals["opinion_supervised_examples"]))
    result: dict[str, object] = {
        "loss": totals["loss"] / examples,
        "bias_loss": totals["bias_loss"] / examples,
        "opinion_loss": totals["opinion_loss"] / examples,
        "opinion_nonfactual_loss": totals["opinion_nonfactual_loss"] / examples,
        "opinion_conditional_loss": totals["opinion_conditional_loss"] / examples,
        "bias_accuracy": totals["bias_correct"] / bias_examples,
        "opinion_accuracy": totals["opinion_correct"] / opinion_examples,
        "bias_macro_f1": bias_metrics["macro_f1"],
        "opinion_macro_f1": opinion_metrics["macro_f1"],
        "opinion_middle_recall": middle_recall,
        "bias_supervised_examples": int(totals["bias_supervised_examples"]),
        "opinion_supervised_examples": int(totals["opinion_supervised_examples"]),
        "conditional_opinion_examples": int(totals["conditional_opinion_examples"]),
        "bias_per_class": bias_metrics["per_class"],
        "opinion_per_class": opinion_metrics["per_class"],
        "bias_confusion_matrix": bias_metrics["confusion_matrix"],
        "opinion_confusion_matrix": opinion_metrics["confusion_matrix"],
    }
    if opinion_head_type == HIERARCHICAL_ORDINAL_HEAD_TYPE:
        hierarchical_diagnostics = hierarchical_opinion_evaluation_diagnostics(
            torch.cat(hierarchical_opinion_outputs, dim=0),
            torch.cat(hierarchical_opinion_labels, dim=0),
            stage_thresholds=hierarchical_stage_thresholds,
        )
        result["hierarchical_opinion_diagnostics"] = hierarchical_diagnostics
        log_hierarchical_opinion_diagnostics(evaluation_name, hierarchical_diagnostics)
    return result


def save_outputs(
    output_dir: Path,
    model: SentenceTransformer,
    bias_head: torch.nn.Module,
    opinion_head: torch.nn.Module,
    args: argparse.Namespace,
    bias_label2id: dict[str, int],
    opinion_label2id: dict[str, int],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    parameter_summary: dict[str, int],
    embedding_dim: int,
    training_history: list[dict[str, object]],
    class_weight_summary: dict[str, Any],
    final_test_metrics: dict[str, object] | None,
    best_validation_summary: dict[str, object] | None,
    foundation_context: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))

    classifier_mode = resolve_classifier_mode(args)
    production_mode = classifier_mode == PRODUCTION_CLASSIFIER_MODE
    if production_mode:
        head_type = INDEPENDENT_BINARY_HEADS_TYPE
        head_architecture = binary_head_architecture(
            head_type=BINARY_OPINION_STYLE_HEAD_TYPE,
            embedding_dim=embedding_dim,
            hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
            logit_name="opinionated_style_logit",
        )
        checkpoint_payload = build_production_classification_heads_checkpoint_payload(
            bias_head=bias_head,
            opinion_style_head=opinion_head,
            embedding_dim=embedding_dim,
            hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
            bias_label2id=bias_label2id,
            opinion_style_label2id=opinion_label2id,
        )
    else:
        head_type = validate_opinion_head_type(args.opinion_head_type)
        head_architecture = opinion_head_architecture(
            head_type=head_type,
            embedding_dim=embedding_dim,
            hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
            opinion_label2id=opinion_label2id,
        )
        checkpoint_payload = build_classification_heads_checkpoint_payload(
            bias_head=bias_head,
            opinion_head=opinion_head,
            embedding_dim=embedding_dim,
            hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
            bias_label2id=bias_label2id,
            opinion_label2id=opinion_label2id,
            head_type=head_type,
        )
    torch.save(
        checkpoint_payload,
        output_dir / "classification_heads.pt",
    )

    (output_dir / "bias_label_mapping.json").write_text(
        json.dumps(bias_label2id, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    opinion_mapping_filename = (
        "opinion_style_label_mapping.json"
        if production_mode
        else "opinion_label_mapping.json"
    )
    (output_dir / opinion_mapping_filename).write_text(
        json.dumps(opinion_label2id, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "model_name": args.model_name,
        "data_dir": str(args.data_dir),
        "data_glob": args.data_glob,
        "train_data_glob": args.train_data_glob,
        "test_data_glob": args.test_data_glob,
        "split_column": args.split_column,
        "train_split_value": args.train_split_value,
        "test_split_value": args.test_split_value,
        "bias_label_column": args.bias_label_column,
        "opinion_label_column": args.opinion_label_column,
        "include_no_agreement": args.include_no_agreement,
        "missing_label_policy": args.missing_label_policy,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "validation_size": args.validation_size,
        "calibration_size": args.calibration_size,
        "test_size": args.test_size,
        "split_strategy": args.split_strategy,
        "group_column": args.group_column,
        "article_id_column": args.article_id_column,
        "event_column": args.event_column,
        "leakage_policy": args.leakage_policy,
        "precision_confidence": args.precision_confidence,
        "allow_legacy_calibration": args.allow_legacy_calibration,
        "class_weighting": args.class_weighting,
        "opinion_sampling": args.opinion_sampling,
        "class_weight_summary": class_weight_summary,
        "max_seq_length": args.max_seq_length,
        "freeze_first_n_layers": args.freeze_first_n_layers,
        "freeze_embeddings": args.freeze_embeddings,
        "head_hidden_dim": args.head_hidden_dim,
        "dropout": args.dropout,
        "checkpoint_schema_version": (
            CLASSIFICATION_HEADS_SCHEMA_VERSION
            if production_mode
            else LEGACY_CLASSIFICATION_HEADS_SCHEMA_VERSION
        ),
        "classifier_input_contract": (
            data_pipeline.CLASSIFIER_INPUT_CONTRACT if production_mode else None
        ),
        "classifier_mode": classifier_mode,
        "legacy_compatibility": not production_mode,
        "production_eligible": bool(
            production_mode and args.checkpoint_selection == "multi_objective"
        ),
        "head_type": head_type,
        "opinion_style_head_architecture" if production_mode else "opinion_head_architecture": head_architecture,
        "opinion_style_loss_weight": args.opinion_style_loss_weight,
        "save_best_checkpoint": args.save_best_checkpoint,
        "early_stopping_patience": args.early_stopping_patience,
        "checkpoint_selection": args.checkpoint_selection,
        "monitor_metric": args.monitor_metric,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "parameter_summary": parameter_summary,
        "embedding_dim": embedding_dim,
        "final_test_metrics": final_test_metrics,
        "best_validation_summary": best_validation_summary,
    }
    if production_mode:
        metadata["bias_head_type"] = BINARY_BIAS_HEAD_TYPE
        metadata["opinion_style_head_type"] = BINARY_OPINION_STYLE_HEAD_TYPE
        metadata["opinion_style_target_mapping"] = dict(
            data_pipeline.SOURCE_OPINION_TIER_TO_STYLE
        )
    else:
        metadata["middle_class_weight"] = args.middle_class_weight
        metadata["opinion_focal_gamma"] = args.opinion_focal_gamma
        metadata["opinion_loss_alpha"] = args.opinion_style_loss_weight
    if foundation_context is not None:
        split_manifest = foundation_context["split_manifest"]
        metadata["split_manifest_path"] = str(foundation_context["split_manifest_path"])
        metadata["split_manifest_sha256"] = split_manifest["split_content_sha256"]
        metadata["canonical_data_manifest_path"] = str(
            foundation_context["canonical_data_manifest_path"]
        )
        metadata["canonical_data_manifest_sha256"] = foundation_context[
            "canonical_data_manifest"
        ]["canonical_data_manifest_sha256"]
        metadata["cleaning_stats"] = foundation_context["cleaning_stats"]
        metadata["calibration_rows"] = len(foundation_context["partitions"]["calibration"])
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(training_history, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    if foundation_context is not None:
        project_root = Path(__file__).resolve().parent
        model_exclusions = (
            "classification_heads.pt",
            "bias_label_mapping.json",
            "opinion_label_mapping.json",
            "opinion_style_label_mapping.json",
            "training_metadata.json",
            "training_history.json",
            "evaluation_scorecard.json",
            "artifact_manifest.json",
            "calibration.json",
        )
        artifact = artifact_manifest.build_artifact_manifest(
            project_root,
            model_path=output_dir,
            heads_path=output_dir / "classification_heads.pt",
            split_manifest_path=foundation_context["split_manifest_path"],
            canonical_data_manifest_path=foundation_context["canonical_data_manifest_path"],
            evaluation_scorecard_path=None,
            code_paths=(
                Path(__file__).resolve(),
                project_root / "data_pipeline.py",
                project_root / "evaluation.py",
                project_root / "artifact_manifest.py",
            ),
            label_mappings=(
                {
                    "bias_label2id": bias_label2id,
                    "opinion_style_label2id": opinion_label2id,
                }
                if production_mode
                else {
                    "bias_label2id": bias_label2id,
                    "opinion_label2id": opinion_label2id,
                }
            ),
            head_type=head_type,
            head_architecture={
                "bias_head": (
                    binary_head_architecture(
                        head_type=BINARY_BIAS_HEAD_TYPE,
                        embedding_dim=embedding_dim,
                        hidden_dim=args.head_hidden_dim,
                        dropout=args.dropout,
                        logit_name="bias_logit",
                    )
                    if production_mode
                    else "one_hidden_layer_mlp"
                ),
                "opinion_style_head" if production_mode else "opinion_head": head_architecture,
                "embedding_dim": embedding_dim,
                "head_hidden_dim": args.head_hidden_dim,
                "dropout": args.dropout,
            },
            training_configuration=metadata,
            source_summary={
                "cleaning_stats": foundation_context["cleaning_stats"],
                "partition_summaries": foundation_context["split_manifest"]["partition_summaries"],
            },
            model_excluded_relative_paths=model_exclusions,
        )
        artifact_manifest.write_artifact_manifest(output_dir / "artifact_manifest.json", artifact)


def is_improved_metric(metric_name: str, current: float, best: float | None) -> bool:
    if best is None:
        return True
    if metric_name == "loss":
        return current < best
    return current > best


def multi_objective_checkpoint_key(
    metrics: Mapping[str, object],
    *,
    classifier_mode: str = PRODUCTION_CLASSIFIER_MODE,
) -> tuple[float, ...]:
    """Return the mode-specific ordered development objectives."""

    if classifier_mode == PRODUCTION_CLASSIFIER_MODE:
        return (
            float(metrics["bias_macro_f1"]),
            float(metrics["bias_biased_precision"]),
            float(metrics["bias_biased_recall"]),
            float(metrics["opinion_style_macro_f1"]),
            float(metrics["opinion_style_objective_recall"]),
            float(metrics["opinion_style_opinionated_recall"]),
            -float(metrics["loss"]),
        )
    if classifier_mode in {
        LEGACY_HIERARCHICAL_CLASSIFIER_MODE,
        LEGACY_FLAT_CLASSIFIER_MODE,
    }:
        return (
            float(metrics["opinion_macro_f1"]),
            float(metrics["opinion_middle_recall"]),
            float(metrics["bias_macro_f1"]),
            -float(metrics["loss"]),
        )
    raise ValueError(f"Unsupported classifier mode: {classifier_mode!r}.")


def is_improved_checkpoint(
    *,
    strategy: str,
    metrics: Mapping[str, object],
    monitor_metric: str,
    best_value: tuple[float, ...] | float | None,
    classifier_mode: str = PRODUCTION_CLASSIFIER_MODE,
) -> tuple[bool, tuple[float, ...] | float]:
    if strategy == "multi_objective":
        current: tuple[float, ...] | float = multi_objective_checkpoint_key(
            metrics, classifier_mode=classifier_mode
        )
        return best_value is None or current > best_value, current
    if strategy == "monitor_metric":
        current = float(metrics[monitor_metric])
        return (
            is_improved_metric(
                monitor_metric,
                current,
                None if best_value is None else float(best_value),
            ),
            current,
        )
    raise ValueError(f"Unsupported checkpoint selection strategy: {strategy!r}.")


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        level=logging.INFO,
    )
    args = parse_args()
    import_training_dependencies()
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = bool(args.fp16 and device.type == "cuda")

    prepared = prepare_data_and_splits(args)
    train_df = prepared["partitions"]["train"]
    val_df = prepared["partitions"]["development"]
    calibration_df = prepared["partitions"]["calibration"]
    test_df = prepared["partitions"]["locked_test"]
    bias_label2id = prepared["bias_label2id"]
    opinion_label2id = prepared["opinion_target_label2id"]
    classifier_mode = prepared["classifier_mode"]
    production_mode = classifier_mode == PRODUCTION_CLASSIFIER_MODE
    LOGGER.info(
        "Split rows: train=%d development=%d calibration=%d locked_test=%d",
        len(train_df),
        len(val_df),
        len(calibration_df),
        len(test_df),
    )

    model = SentenceTransformer(args.model_name, device=str(device))
    model.max_seq_length = args.max_seq_length
    parameter_summary = freeze_mpnet_layers(
        model,
        first_n_layers=args.freeze_first_n_layers,
        freeze_embeddings=args.freeze_embeddings,
    )

    embedding_dim = model.get_sentence_embedding_dimension()
    if embedding_dim is None:
        raise ValueError("Could not infer sentence embedding dimension from the model.")

    if production_mode:
        bias_head = build_binary_head(
            embedding_dim,
            args.head_hidden_dim,
            args.dropout,
            head_type=BINARY_BIAS_HEAD_TYPE,
        ).to(device)
        opinion_head = build_binary_head(
            embedding_dim,
            args.head_hidden_dim,
            args.dropout,
            head_type=BINARY_OPINION_STYLE_HEAD_TYPE,
        ).to(device)
    else:
        bias_head = build_mlp_head(
            embedding_dim=embedding_dim,
            hidden_dim=args.head_hidden_dim,
            num_labels=len(bias_label2id),
            dropout=args.dropout,
        ).to(device)
        opinion_head = build_opinion_head(
            embedding_dim=embedding_dim,
            hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
            opinion_label2id=opinion_label2id,
            head_type=args.opinion_head_type,
        ).to(device)

    LOGGER.info(
        "SBERT parameters: total=%d trainable=%d frozen=%d",
        parameter_summary["total"],
        parameter_summary["trainable"],
        parameter_summary["frozen"],
    )

    train_sampler = (
        None if production_mode else build_opinion_balanced_sampler(train_df, args)
    )
    dataset_class = BiasOpinionStyleDataset if production_mode else BiasOpinionDataset
    collate_function = collate_opinion_style_batch if production_mode else collate_batch
    train_dataloader = DataLoader(
        dataset_class(train_df),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_function,
    )
    val_dataloader = DataLoader(
        dataset_class(val_df),
        shuffle=False,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_function,
    )
    test_dataloader = DataLoader(
        dataset_class(test_df),
        shuffle=False,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_function,
    )
    if len(train_dataloader) == 0:
        raise ValueError(
            "Training dataloader is empty. Reduce --batch-size or add training data."
        )

    bias_class_weights = None
    if production_mode:
        (
            bias_criterion,
            opinion_criterion,
            bias_class_weights,
            opinion_class_weights,
            class_weight_summary,
        ) = build_production_loss_functions(train_df, args, device)
    else:
        (
            bias_criterion,
            opinion_criterion,
            opinion_class_weights,
            class_weight_summary,
        ) = build_loss_functions(
            train_df=train_df,
            args=args,
            device=device,
            bias_num_labels=len(bias_label2id),
            opinion_num_labels=len(opinion_label2id),
        )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(bias_head.parameters()) + list(opinion_head.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_training_steps = len(train_dataloader) * args.epochs
    scheduler = build_linear_warmup_decay_scheduler(
        optimizer=optimizer,
        total_steps=total_training_steps,
        warmup_ratio=args.warmup_ratio,
    )

    LOGGER.info(
        "Starting %s fine-tuning: total_loss = bias_loss + %.3f * %s",
        classifier_mode,
        args.opinion_style_loss_weight,
        "opinion_style_loss" if production_mode else "legacy_opinion_loss",
    )
    LOGGER.info(
        "Regularization: dropout=%.3f, AdamW weight_decay=%.3f",
        args.dropout,
        args.weight_decay,
    )
    LOGGER.info("Class weighting: %s", args.class_weighting)
    if not production_mode:
        LOGGER.info("Legacy opinion sampling: %s", args.opinion_sampling)
    if class_weight_summary["bias"] is not None:
        LOGGER.info("Bias class weights: %s", class_weight_summary["bias"])
        LOGGER.info(
            "%s class weights: %s",
            "Opinion style" if production_mode else "Legacy opinion",
            class_weight_summary[
                "opinion_style" if production_mode else "opinion"
            ],
        )
    LOGGER.info(
        "LR scheduler: linear warmup/decay over %d step(s), warmup_ratio=%.3f",
        total_training_steps,
        args.warmup_ratio,
    )

    training_history: list[dict[str, object]] = []
    best_validation_summary: dict[str, object] | None = None
    best_selection_value: tuple[float, ...] | float | None = None
    epochs_without_improvement = 0
    best_checkpoint_dir = args.output_dir / "best"
    for epoch_idx in range(args.epochs):
        if production_mode:
            train_metrics = train_production_one_epoch(
                model=model,
                bias_head=bias_head,
                opinion_style_head=opinion_head,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                bias_criterion=bias_criterion,
                opinion_style_criterion=opinion_criterion,
                bias_class_weights=bias_class_weights,
                opinion_style_class_weights=opinion_class_weights,
                opinion_style_loss_weight=args.opinion_style_loss_weight,
                device=device,
                use_amp=use_amp,
                max_grad_norm=args.max_grad_norm,
            )
        else:
            train_metrics = train_one_epoch(
                model=model,
                bias_head=bias_head,
                opinion_head=opinion_head,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                bias_criterion=bias_criterion,
                opinion_criterion=opinion_criterion,
                opinion_class_weights=opinion_class_weights,
                opinion_loss_alpha=args.opinion_style_loss_weight,
                opinion_head_type=args.opinion_head_type,
                middle_class_weight=args.middle_class_weight,
                opinion_focal_gamma=args.opinion_focal_gamma,
                device=device,
                use_amp=use_amp,
                max_grad_norm=args.max_grad_norm,
            )
        epoch_record: dict[str, object] = {
            "epoch": epoch_idx + 1,
            "train": train_metrics,
        }

        if not args.skip_validation and not val_df.empty:
            if production_mode:
                val_metrics = evaluate_production(
                    model=model,
                    bias_head=bias_head,
                    opinion_style_head=opinion_head,
                    dataloader=val_dataloader,
                    bias_criterion=bias_criterion,
                    opinion_style_criterion=opinion_criterion,
                    bias_class_weights=bias_class_weights,
                    opinion_style_class_weights=opinion_class_weights,
                    opinion_style_loss_weight=args.opinion_style_loss_weight,
                    device=device,
                    use_amp=use_amp,
                )
            else:
                val_metrics = evaluate(
                    model=model,
                    bias_head=bias_head,
                    opinion_head=opinion_head,
                    dataloader=val_dataloader,
                    bias_criterion=bias_criterion,
                    opinion_criterion=opinion_criterion,
                    opinion_class_weights=opinion_class_weights,
                    opinion_loss_alpha=args.opinion_style_loss_weight,
                    opinion_head_type=args.opinion_head_type,
                    middle_class_weight=args.middle_class_weight,
                    opinion_focal_gamma=args.opinion_focal_gamma,
                    device=device,
                    use_amp=use_amp,
                    bias_num_labels=len(bias_label2id),
                    opinion_num_labels=len(opinion_label2id),
                    evaluation_name=f"validation epoch {epoch_idx + 1}",
                )
            epoch_record["validation"] = val_metrics
            if production_mode:
                LOGGER.info(
                    "Epoch %d/%d train_loss=%.4f dev_loss=%.4f "
                    "bias_mF1=%.4f biased_P/R=%.4f/%.4f "
                    "style_mF1=%.4f objective/opinionated_R=%.4f/%.4f",
                    epoch_idx + 1,
                    args.epochs,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["bias_macro_f1"],
                    val_metrics["bias_biased_precision"],
                    val_metrics["bias_biased_recall"],
                    val_metrics["opinion_style_macro_f1"],
                    val_metrics["opinion_style_objective_recall"],
                    val_metrics["opinion_style_opinionated_recall"],
                )
            else:
                LOGGER.info(
                    "Epoch %d/%d train_loss=%.4f dev_loss=%.4f "
                    "legacy_bias_mF1=%.4f legacy_opinion_mF1=%.4f "
                    "legacy_middle_R=%.4f",
                    epoch_idx + 1,
                    args.epochs,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["bias_macro_f1"],
                    val_metrics["opinion_macro_f1"],
                    val_metrics["opinion_middle_recall"],
                )
        else:
            LOGGER.info(
                "Epoch %d/%d train_loss=%.4f bias_loss=%.4f %s=%.4f",
                epoch_idx + 1,
                args.epochs,
                train_metrics["loss"],
                train_metrics["bias_loss"],
                "opinion_style_loss" if production_mode else "legacy_opinion_loss",
                train_metrics[
                    "opinion_style_loss" if production_mode else "opinion_loss"
                ],
            )

        training_history.append(epoch_record)

        if "validation" in epoch_record:
            validation_metrics = epoch_record["validation"]
            improved, current_selection_value = is_improved_checkpoint(
                strategy=args.checkpoint_selection,
                metrics=validation_metrics,
                monitor_metric=args.monitor_metric,
                best_value=best_selection_value,
                classifier_mode=classifier_mode,
            )
            if improved:
                best_selection_value = current_selection_value
                epochs_without_improvement = 0
                best_validation_summary = {
                    "epoch": epoch_idx + 1,
                    "strategy": args.checkpoint_selection,
                    "monitor_metric": (
                        args.monitor_metric
                        if args.checkpoint_selection == "monitor_metric"
                        else None
                    ),
                    "selection_value": current_selection_value,
                    "development_metrics": (
                        {
                            "bias_macro_f1": validation_metrics["bias_macro_f1"],
                            "bias_biased_precision": validation_metrics[
                                "bias_biased_precision"
                            ],
                            "bias_biased_recall": validation_metrics[
                                "bias_biased_recall"
                            ],
                            "opinion_style_macro_f1": validation_metrics[
                                "opinion_style_macro_f1"
                            ],
                            "opinion_style_objective_recall": validation_metrics[
                                "opinion_style_objective_recall"
                            ],
                            "opinion_style_opinionated_recall": validation_metrics[
                                "opinion_style_opinionated_recall"
                            ],
                            "loss": validation_metrics["loss"],
                        }
                        if production_mode
                        else {
                            "opinion_macro_f1": validation_metrics[
                                "opinion_macro_f1"
                            ],
                            "opinion_middle_recall": validation_metrics[
                                "opinion_middle_recall"
                            ],
                            "bias_macro_f1": validation_metrics["bias_macro_f1"],
                            "loss": validation_metrics["loss"],
                        }
                    ),
                    "checkpoint_dir": str(best_checkpoint_dir),
                }
                if args.save_best_checkpoint:
                    save_outputs(
                        output_dir=best_checkpoint_dir,
                        model=model,
                        bias_head=bias_head,
                        opinion_head=opinion_head,
                        args=args,
                        bias_label2id=bias_label2id,
                        opinion_label2id=opinion_label2id,
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        parameter_summary=parameter_summary,
                        embedding_dim=embedding_dim,
                        training_history=training_history,
                        class_weight_summary=class_weight_summary,
                        final_test_metrics=None,
                        best_validation_summary=best_validation_summary,
                        foundation_context=prepared,
                    )
                    LOGGER.info(
                        "Saved best validation checkpoint to %s using %s=%s",
                        best_checkpoint_dir,
                        args.checkpoint_selection,
                        current_selection_value,
                    )
            else:
                epochs_without_improvement += 1
                LOGGER.info(
                    "Validation %s did not improve for %d epoch(s)",
                    args.checkpoint_selection,
                    epochs_without_improvement,
                )

            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                LOGGER.info(
                    "Early stopping after %d epoch(s) without validation improvement",
                    epochs_without_improvement,
                )
                break

    final_test_metrics = None
    if args.evaluate_test_final and not test_df.empty:
        if production_mode:
            final_test_metrics = evaluate_production(
                model=model,
                bias_head=bias_head,
                opinion_style_head=opinion_head,
                dataloader=test_dataloader,
                bias_criterion=bias_criterion,
                opinion_style_criterion=opinion_criterion,
                bias_class_weights=bias_class_weights,
                opinion_style_class_weights=opinion_class_weights,
                opinion_style_loss_weight=args.opinion_style_loss_weight,
                device=device,
                use_amp=use_amp,
            )
            LOGGER.info(
                "Final test loss=%.4f bias_mF1=%.4f opinion_style_mF1=%.4f",
                final_test_metrics["loss"],
                final_test_metrics["bias_macro_f1"],
                final_test_metrics["opinion_style_macro_f1"],
            )
        else:
            final_test_metrics = evaluate(
                model=model,
                bias_head=bias_head,
                opinion_head=opinion_head,
                dataloader=test_dataloader,
                bias_criterion=bias_criterion,
                opinion_criterion=opinion_criterion,
                opinion_class_weights=opinion_class_weights,
                opinion_loss_alpha=args.opinion_style_loss_weight,
                opinion_head_type=args.opinion_head_type,
                middle_class_weight=args.middle_class_weight,
                opinion_focal_gamma=args.opinion_focal_gamma,
                device=device,
                use_amp=use_amp,
                bias_num_labels=len(bias_label2id),
                opinion_num_labels=len(opinion_label2id),
                evaluation_name="final legacy test",
            )
            LOGGER.info(
                "Final legacy test loss=%.4f bias_mF1=%.4f opinion_mF1=%.4f",
                final_test_metrics["loss"],
                final_test_metrics["bias_macro_f1"],
                final_test_metrics["opinion_macro_f1"],
            )

    save_outputs(
        output_dir=args.output_dir,
        model=model,
        bias_head=bias_head,
        opinion_head=opinion_head,
        args=args,
        bias_label2id=bias_label2id,
        opinion_label2id=opinion_label2id,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        parameter_summary=parameter_summary,
        embedding_dim=embedding_dim,
        training_history=training_history,
        class_weight_summary=class_weight_summary,
        final_test_metrics=final_test_metrics,
        best_validation_summary=best_validation_summary,
        foundation_context=prepared,
    )
    LOGGER.info("Fine-tuned model and classification heads saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
