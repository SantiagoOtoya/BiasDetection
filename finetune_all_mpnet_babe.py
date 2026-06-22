#!/usr/bin/env python
"""Fine-tune all-mpnet-base-v2 SBERT on BABE media-bias labels.

The default setup uses BABE_HF when present, otherwise the curated BABE final
label CSV files in ``data``. If BABE is already split into train/test files,
the script keeps test separate and creates validation only from the train split.
It drops ``No agreement`` rows,
freezes MPNet encoder layers 0-5, and fine-tunes the last six encoder layers
with two one-hidden-layer MLP classification heads:

    total_loss = bias_loss + alpha * opinion_loss

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
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger("finetune_all_mpnet_babe")
SUPPORTED_DATA_SUFFIXES = {".csv", ".parquet"}
np = None
pd = None
torch = None
SentenceTransformer = None
DataLoader = None


def import_training_dependencies() -> None:
    global DataLoader, SentenceTransformer, np, pd, torch

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
        from torch.utils.data import DataLoader as data_loader_class
    except ImportError:
        if "torch" not in missing:
            missing.append("torch")
    else:
        DataLoader = data_loader_class

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
        help="Column used for the auxiliary opinion classification loss.",
    )
    parser.add_argument(
        "--include-no-agreement",
        action="store_true",
        help="Keep rows labeled 'No agreement' instead of dropping them.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable text-level deduplication and conflict removal.",
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
        help="Validation fraction. Set to 0 to skip validation metrics.",
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
        "--opinion-loss-alpha",
        type=float,
        default=0.3,
        help="Auxiliary opinion-loss weight in bias_loss + alpha * opinion_loss.",
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default="balanced",
        help=(
            "Use inverse-frequency class weights for both heads. Recommended "
            "for the opinion task because its labels are less balanced."
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
        "--monitor-metric",
        default="bias_macro_f1",
        choices=["loss", "bias_accuracy", "bias_macro_f1", "opinion_accuracy", "opinion_macro_f1"],
        help="Validation metric used for best-checkpoint selection.",
    )
    return parser.parse_args()


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

    stratify_key = (
        df["bias_label_id"].astype(str) + "::" + df["opinion_label_id"].astype(str)
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
    def __init__(self, df: pd.DataFrame) -> None:
        self.texts = df["text"].tolist()
        self.bias_labels = df["bias_label_id"].astype(int).tolist()
        self.opinion_labels = df["opinion_label_id"].astype(int).tolist()

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
    counts = np.bincount(labels.astype(int).to_numpy(), minlength=num_labels)
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
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, list[float] | None]]:
    bias_weights = None
    opinion_weights = None
    if args.class_weighting == "balanced":
        bias_weights = build_class_weights(
            train_df["bias_label_id"], bias_num_labels, device
        )
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
    }
    return (
        torch.nn.CrossEntropyLoss(weight=bias_weights),
        torch.nn.CrossEntropyLoss(weight=opinion_weights),
        weight_summary,
    )


def classification_metrics(
    predictions: list[int], labels: list[int], num_labels: int
) -> dict[str, object]:
    if not labels:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": []}

    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    per_class: list[dict[str, float | int]] = []
    f1_scores: list[float] = []
    for label_id in range(num_labels):
        true_positive = sum(
            int(pred == label_id and label == label_id)
            for pred, label in zip(predictions, labels)
        )
        false_positive = sum(
            int(pred == label_id and label != label_id)
            for pred, label in zip(predictions, labels)
        )
        false_negative = sum(
            int(pred != label_id and label == label_id)
            for pred, label in zip(predictions, labels)
        )
        support = sum(int(label == label_id) for label in labels)

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_scores.append(f1)
        per_class.append(
            {
                "label_id": label_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    return {
        "accuracy": correct / len(labels),
        "macro_f1": sum(f1_scores) / max(1, num_labels),
        "per_class": per_class,
    }


def sentence_embeddings(
    model: SentenceTransformer, texts: list[str], device: torch.device
) -> torch.Tensor:
    features = model.tokenize(texts)
    features = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in features.items()
    }
    return model(features)["sentence_embedding"]


def batch_loss(
    model: SentenceTransformer,
    bias_head: torch.nn.Module,
    opinion_head: torch.nn.Module,
    batch: dict[str, object],
    bias_criterion: torch.nn.Module,
    opinion_criterion: torch.nn.Module,
    opinion_loss_alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = sentence_embeddings(model, batch["texts"], device)
    bias_logits = bias_head(embeddings)
    opinion_logits = opinion_head(embeddings)
    bias_labels = batch["bias_labels"].to(device)
    opinion_labels = batch["opinion_labels"].to(device)

    bias_loss = bias_criterion(bias_logits, bias_labels)
    opinion_loss = opinion_criterion(opinion_logits, opinion_labels)
    total_loss = bias_loss + opinion_loss_alpha * opinion_loss
    return total_loss, bias_loss, opinion_loss, bias_logits, opinion_logits


def train_one_epoch(
    model: SentenceTransformer,
    bias_head: torch.nn.Module,
    opinion_head: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    bias_criterion: torch.nn.Module,
    opinion_criterion: torch.nn.Module,
    opinion_loss_alpha: float,
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
        "examples": 0,
    }

    for batch in dataloader:
        batch_size = len(batch["texts"])
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            total_loss, bias_loss, opinion_loss, _, _ = batch_loss(
                model=model,
                bias_head=bias_head,
                opinion_head=opinion_head,
                batch=batch,
                bias_criterion=bias_criterion,
                opinion_criterion=opinion_criterion,
                opinion_loss_alpha=opinion_loss_alpha,
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
        totals["examples"] += batch_size

    examples = max(1, totals["examples"])
    return {
        "loss": totals["loss"] / examples,
        "bias_loss": totals["bias_loss"] / examples,
        "opinion_loss": totals["opinion_loss"] / examples,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }


def evaluate(
    model: SentenceTransformer,
    bias_head: torch.nn.Module,
    opinion_head: torch.nn.Module,
    dataloader: DataLoader,
    bias_criterion: torch.nn.Module,
    opinion_criterion: torch.nn.Module,
    opinion_loss_alpha: float,
    device: torch.device,
    use_amp: bool,
    bias_num_labels: int,
    opinion_num_labels: int,
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
        "bias_correct": 0,
        "opinion_correct": 0,
        "examples": 0,
    }
    bias_predictions: list[int] = []
    bias_targets: list[int] = []
    opinion_predictions: list[int] = []
    opinion_targets: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            batch_size = len(batch["texts"])
            with torch.cuda.amp.autocast(enabled=use_amp):
                total_loss, bias_loss, opinion_loss, bias_logits, opinion_logits = batch_loss(
                    model=model,
                    bias_head=bias_head,
                    opinion_head=opinion_head,
                    batch=batch,
                    bias_criterion=bias_criterion,
                    opinion_criterion=opinion_criterion,
                    opinion_loss_alpha=opinion_loss_alpha,
                    device=device,
                )

            bias_labels = batch["bias_labels"].to(device)
            opinion_labels = batch["opinion_labels"].to(device)
            batch_bias_predictions = bias_logits.argmax(dim=-1)
            batch_opinion_predictions = opinion_logits.argmax(dim=-1)
            totals["loss"] += float(total_loss.cpu()) * batch_size
            totals["bias_loss"] += float(bias_loss.cpu()) * batch_size
            totals["opinion_loss"] += float(opinion_loss.cpu()) * batch_size
            totals["bias_correct"] += int(
                (batch_bias_predictions == bias_labels).sum().cpu()
            )
            totals["opinion_correct"] += int(
                (batch_opinion_predictions == opinion_labels).sum().cpu()
            )
            totals["examples"] += batch_size
            bias_predictions.extend(batch_bias_predictions.cpu().tolist())
            bias_targets.extend(bias_labels.cpu().tolist())
            opinion_predictions.extend(batch_opinion_predictions.cpu().tolist())
            opinion_targets.extend(opinion_labels.cpu().tolist())

    examples = max(1, totals["examples"])
    bias_metrics = classification_metrics(
        bias_predictions, bias_targets, bias_num_labels
    )
    opinion_metrics = classification_metrics(
        opinion_predictions, opinion_targets, opinion_num_labels
    )
    return {
        "loss": totals["loss"] / examples,
        "bias_loss": totals["bias_loss"] / examples,
        "opinion_loss": totals["opinion_loss"] / examples,
        "bias_accuracy": totals["bias_correct"] / examples,
        "opinion_accuracy": totals["opinion_correct"] / examples,
        "bias_macro_f1": bias_metrics["macro_f1"],
        "opinion_macro_f1": opinion_metrics["macro_f1"],
        "bias_per_class": bias_metrics["per_class"],
        "opinion_per_class": opinion_metrics["per_class"],
    }


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
    class_weight_summary: dict[str, list[float] | None],
    final_test_metrics: dict[str, object] | None,
    best_validation_summary: dict[str, object] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))

    torch.save(
        {
            "bias_head_state_dict": bias_head.state_dict(),
            "opinion_head_state_dict": opinion_head.state_dict(),
            "embedding_dim": embedding_dim,
            "head_hidden_dim": args.head_hidden_dim,
            "dropout": args.dropout,
            "bias_label2id": bias_label2id,
            "opinion_label2id": opinion_label2id,
        },
        output_dir / "classification_heads.pt",
    )

    (output_dir / "bias_label_mapping.json").write_text(
        json.dumps(bias_label2id, indent=2), encoding="utf-8"
    )
    (output_dir / "opinion_label_mapping.json").write_text(
        json.dumps(opinion_label2id, indent=2), encoding="utf-8"
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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "class_weighting": args.class_weighting,
        "class_weight_summary": class_weight_summary,
        "max_seq_length": args.max_seq_length,
        "freeze_first_n_layers": args.freeze_first_n_layers,
        "freeze_embeddings": args.freeze_embeddings,
        "head_hidden_dim": args.head_hidden_dim,
        "dropout": args.dropout,
        "opinion_loss_alpha": args.opinion_loss_alpha,
        "save_best_checkpoint": args.save_best_checkpoint,
        "early_stopping_patience": args.early_stopping_patience,
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
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(training_history, indent=2), encoding="utf-8"
    )


def is_improved_metric(metric_name: str, current: float, best: float | None) -> bool:
    if best is None:
        return True
    if metric_name == "loss":
        return current < best
    return current > best


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

    train_pool_df, test_df, bias_label2id, opinion_label2id = load_dataset(args)
    train_df, val_df = split_dataset(train_pool_df, args.validation_size, args.seed)
    LOGGER.info(
        "Split rows: train=%d validation=%d test=%d",
        len(train_df),
        len(val_df),
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

    bias_head = build_mlp_head(
        embedding_dim=embedding_dim,
        hidden_dim=args.head_hidden_dim,
        num_labels=len(bias_label2id),
        dropout=args.dropout,
    ).to(device)
    opinion_head = build_mlp_head(
        embedding_dim=embedding_dim,
        hidden_dim=args.head_hidden_dim,
        num_labels=len(opinion_label2id),
        dropout=args.dropout,
    ).to(device)

    LOGGER.info(
        "SBERT parameters: total=%d trainable=%d frozen=%d",
        parameter_summary["total"],
        parameter_summary["trainable"],
        parameter_summary["frozen"],
    )

    train_dataloader = DataLoader(
        BiasOpinionDataset(train_df),
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_dataloader = DataLoader(
        BiasOpinionDataset(val_df),
        shuffle=False,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    test_dataloader = DataLoader(
        BiasOpinionDataset(test_df),
        shuffle=False,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    if len(train_dataloader) == 0:
        raise ValueError(
            "Training dataloader is empty. Reduce --batch-size or add training data."
        )

    bias_criterion, opinion_criterion, class_weight_summary = build_loss_functions(
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
        "Starting fine-tuning: total_loss = bias_loss + %.3f * opinion_loss",
        args.opinion_loss_alpha,
    )
    LOGGER.info(
        "Regularization: dropout=%.3f, AdamW weight_decay=%.3f",
        args.dropout,
        args.weight_decay,
    )
    LOGGER.info("Class weighting: %s", args.class_weighting)
    if class_weight_summary["bias"] is not None:
        LOGGER.info("Bias class weights: %s", class_weight_summary["bias"])
        LOGGER.info("Opinion class weights: %s", class_weight_summary["opinion"])
    LOGGER.info(
        "LR scheduler: linear warmup/decay over %d step(s), warmup_ratio=%.3f",
        total_training_steps,
        args.warmup_ratio,
    )

    training_history: list[dict[str, object]] = []
    best_validation_summary: dict[str, object] | None = None
    best_monitor_score: float | None = None
    epochs_without_improvement = 0
    best_checkpoint_dir = args.output_dir / "best"
    for epoch_idx in range(args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            bias_head=bias_head,
            opinion_head=opinion_head,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            bias_criterion=bias_criterion,
            opinion_criterion=opinion_criterion,
            opinion_loss_alpha=args.opinion_loss_alpha,
            device=device,
            use_amp=use_amp,
            max_grad_norm=args.max_grad_norm,
        )
        epoch_record: dict[str, object] = {
            "epoch": epoch_idx + 1,
            "train": train_metrics,
        }

        if not args.skip_validation and not val_df.empty:
            val_metrics = evaluate(
                model=model,
                bias_head=bias_head,
                opinion_head=opinion_head,
                dataloader=val_dataloader,
                bias_criterion=bias_criterion,
                opinion_criterion=opinion_criterion,
                opinion_loss_alpha=args.opinion_loss_alpha,
                device=device,
                use_amp=use_amp,
                bias_num_labels=len(bias_label2id),
                opinion_num_labels=len(opinion_label2id),
            )
            epoch_record["validation"] = val_metrics
            LOGGER.info(
                "Epoch %d/%d train_loss=%.4f val_loss=%.4f "
                "val_bias_acc=%.4f val_bias_macro_f1=%.4f "
                "val_opinion_acc=%.4f val_opinion_macro_f1=%.4f",
                epoch_idx + 1,
                args.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["bias_accuracy"],
                val_metrics["bias_macro_f1"],
                val_metrics["opinion_accuracy"],
                val_metrics["opinion_macro_f1"],
            )
        else:
            LOGGER.info(
                "Epoch %d/%d train_loss=%.4f bias_loss=%.4f opinion_loss=%.4f",
                epoch_idx + 1,
                args.epochs,
                train_metrics["loss"],
                train_metrics["bias_loss"],
                train_metrics["opinion_loss"],
            )

        training_history.append(epoch_record)

        if "validation" in epoch_record:
            current_monitor_score = float(epoch_record["validation"][args.monitor_metric])
            if is_improved_metric(
                args.monitor_metric, current_monitor_score, best_monitor_score
            ):
                best_monitor_score = current_monitor_score
                epochs_without_improvement = 0
                best_validation_summary = {
                    "epoch": epoch_idx + 1,
                    "metric": args.monitor_metric,
                    "value": current_monitor_score,
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
                    )
                    LOGGER.info(
                        "Saved best validation checkpoint to %s using %s=%.4f",
                        best_checkpoint_dir,
                        args.monitor_metric,
                        current_monitor_score,
                    )
            else:
                epochs_without_improvement += 1
                LOGGER.info(
                    "Validation %s did not improve for %d epoch(s)",
                    args.monitor_metric,
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
        final_test_metrics = evaluate(
            model=model,
            bias_head=bias_head,
            opinion_head=opinion_head,
            dataloader=test_dataloader,
            bias_criterion=bias_criterion,
            opinion_criterion=opinion_criterion,
            opinion_loss_alpha=args.opinion_loss_alpha,
            device=device,
            use_amp=use_amp,
            bias_num_labels=len(bias_label2id),
            opinion_num_labels=len(opinion_label2id),
        )
        LOGGER.info(
            "Final test loss=%.4f test_bias_acc=%.4f test_bias_macro_f1=%.4f "
            "test_opinion_acc=%.4f test_opinion_macro_f1=%.4f",
            final_test_metrics["loss"],
            final_test_metrics["bias_accuracy"],
            final_test_metrics["bias_macro_f1"],
            final_test_metrics["opinion_accuracy"],
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
    )
    LOGGER.info("Fine-tuned model and classification heads saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
