#!/usr/bin/env python
"""Fit validation-based calibration for the SBERT bias/opinion heads."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
LLM_DIR = PROJECT_ROOT / "LLM-inference"
for import_path in (PROJECT_ROOT, LLM_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import finetune_all_mpnet_babe as training
import infer_bias_llm

LOGGER = logging.getLogger("calibrate_sbert_heads")


def parse_args() -> argparse.Namespace:
    default_sbert_dir = PROJECT_ROOT / "models" / "all-mpnet-base-v2-babe" / "best"
    parser = argparse.ArgumentParser(
        description="Fit temperature scaling and high-precision thresholds for SBERT heads."
    )
    parser.add_argument("--sbert-model-dir", type=Path, default=default_sbert_dir)
    parser.add_argument("--classification-heads", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--target-precision", type=float, default=0.90)
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
        "--include-no-agreement",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether rows labeled No agreement are kept.",
    )
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--validation-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def apply_metadata_defaults(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = args.sbert_model_dir / "training_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if args.data_dir is None:
        args.data_dir = resolve_project_path(metadata.get("data_dir", "BABE_HF"))
    if args.data_glob is None:
        args.data_glob = metadata.get("data_glob") or ["final_labels_*.csv"]
    if args.train_data_glob is None:
        args.train_data_glob = metadata.get("train_data_glob")
    if args.test_data_glob is None:
        args.test_data_glob = metadata.get("test_data_glob")
    if args.split_column is None:
        args.split_column = metadata.get("split_column")
    if args.train_split_value is None:
        args.train_split_value = metadata.get("train_split_value", "train")
    if args.test_split_value is None:
        args.test_split_value = metadata.get("test_split_value", "test")
    if args.text_column is None:
        args.text_column = metadata.get("text_column", "text")
    if args.bias_label_column is None:
        args.bias_label_column = metadata.get("bias_label_column", "label_bias")
    if args.opinion_label_column is None:
        args.opinion_label_column = metadata.get("opinion_label_column", "label_opinion")
    if args.include_no_agreement is None:
        args.include_no_agreement = bool(metadata.get("include_no_agreement", False))
    if args.validation_size is None:
        args.validation_size = float(metadata.get("validation_size", 0.15))
    if args.seed is None:
        args.seed = int(metadata.get("seed", 42))
    if args.batch_size is None:
        args.batch_size = int(metadata.get("batch_size", 16))
    if args.output_file is None:
        args.output_file = args.sbert_model_dir / "calibration.json"
    return metadata


def choose_threshold_for_precision(
    scores: Sequence[float],
    labels: Sequence[bool],
    candidate_mask: Sequence[bool],
    target_precision: float,
) -> dict[str, Any]:
    if not 0.0 <= target_precision <= 1.0:
        raise ValueError("target_precision must be between 0 and 1.")
    if not (len(scores) == len(labels) == len(candidate_mask)):
        raise ValueError("scores, labels, and candidate_mask must have the same length.")
    positive_support = sum(1 for label in labels if label)
    if positive_support == 0:
        return {
            "enabled": False,
            "threshold": None,
            "target_precision": target_precision,
            "validation_precision": None,
            "validation_recall": 0.0,
            "validation_selected_count": 0,
            "validation_positive_support": 0,
            "disabled_reason": "no_positive_validation_examples",
        }
    candidate_indices = [index for index, is_candidate in enumerate(candidate_mask) if is_candidate]
    if not candidate_indices:
        return {
            "enabled": False,
            "threshold": None,
            "target_precision": target_precision,
            "validation_precision": None,
            "validation_recall": 0.0,
            "validation_selected_count": 0,
            "validation_positive_support": positive_support,
            "disabled_reason": "no_predicted_positive_validation_examples",
        }
    best: dict[str, Any] | None = None
    best_possible_precision = 0.0
    for threshold in sorted({float(scores[index]) for index in candidate_indices}, reverse=True):
        selected_indices = [
            index for index in candidate_indices if float(scores[index]) >= threshold
        ]
        true_positive = sum(1 for index in selected_indices if labels[index])
        selected_count = len(selected_indices)
        precision = true_positive / selected_count if selected_count else 0.0
        recall = true_positive / positive_support if positive_support else 0.0
        best_possible_precision = max(best_possible_precision, precision)
        if precision < target_precision:
            continue
        candidate = {
            "enabled": True,
            "threshold": threshold,
            "target_precision": target_precision,
            "validation_precision": precision,
            "validation_recall": recall,
            "validation_selected_count": selected_count,
            "validation_positive_support": positive_support,
            "disabled_reason": None,
        }
        if best is None or recall > float(best["validation_recall"]):
            best = candidate
        elif best is not None and recall == best["validation_recall"] and threshold < best["threshold"]:
            best = candidate
    if best is not None:
        return best
    return {
        "enabled": False,
        "threshold": None,
        "target_precision": target_precision,
        "validation_precision": best_possible_precision,
        "validation_recall": 0.0,
        "validation_selected_count": 0,
        "validation_positive_support": positive_support,
        "disabled_reason": "target_precision_not_met",
    }


def fit_temperature(logits: Any, labels: Any) -> float:
    torch = training.torch
    assert torch is not None
    if logits.numel() == 0:
        return 1.0
    logits = logits.float()
    labels = labels.long()
    baseline_nll = cross_entropy_at_temperature(logits, labels, 1.0)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100)

    def closure() -> Any:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 10.0)
        loss = torch.nn.functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError as exc:
        LOGGER.warning("Temperature optimization failed; using temperature=1.0: %s", exc)
        return 1.0

    temperature = float(torch.exp(log_temperature).clamp(0.05, 10.0).detach().cpu().item())
    calibrated_nll = cross_entropy_at_temperature(logits, labels, temperature)
    if calibrated_nll > baseline_nll:
        LOGGER.warning(
            "Temperature %.6f worsened NLL from %.6f to %.6f; using temperature=1.0",
            temperature,
            baseline_nll,
            calibrated_nll,
        )
        return 1.0
    return temperature


def cross_entropy_at_temperature(logits: Any, labels: Any, temperature: float) -> float:
    torch = training.torch
    assert torch is not None
    return float(
        torch.nn.functional.cross_entropy(
            logits.float() / float(temperature), labels.long()
        )
        .detach()
        .cpu()
        .item()
    )


def collect_validation_logits(
    model: Any,
    bias_head: Any,
    opinion_head: Any,
    val_df: Any,
    batch_size: int,
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    torch = training.torch
    assert torch is not None
    dataloader = training.DataLoader(
        training.BiasOpinionDataset(val_df),
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=training.collate_batch,
    )
    bias_logits: list[Any] = []
    opinion_logits: list[Any] = []
    bias_labels: list[Any] = []
    opinion_labels: list[Any] = []
    model.eval()
    bias_head.eval()
    opinion_head.eval()
    with torch.no_grad():
        for batch in dataloader:
            embeddings = training.sentence_embeddings(model, batch["texts"], device)
            bias_logits.append(bias_head(embeddings).detach().cpu())
            opinion_logits.append(opinion_head(embeddings).detach().cpu())
            bias_labels.append(batch["bias_labels"].detach().cpu())
            opinion_labels.append(batch["opinion_labels"].detach().cpu())
    return (
        torch.cat(bias_logits),
        torch.cat(opinion_logits),
        torch.cat(bias_labels),
        torch.cat(opinion_labels),
    )


def calibrate_bias_head(
    logits: Any,
    labels: Any,
    label2id: dict[str, int],
    target_precision: float,
) -> dict[str, Any]:
    torch = training.torch
    assert torch is not None
    if "Biased" not in label2id:
        raise ValueError("Bias labels must contain 'Biased'.")
    positive_id = int(label2id["Biased"])
    temperature = fit_temperature(logits, labels)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    predicted_ids = probabilities.argmax(dim=-1)
    scores = probabilities[:, positive_id].tolist()
    truth = (labels == positive_id).tolist()
    candidates = (predicted_ids == positive_id).tolist()
    result = choose_threshold_for_precision(scores, truth, candidates, target_precision)
    result.update(
        {
            "temperature": temperature,
            "score": "P(Biased)",
            "positive_label": "Biased",
            "nll_before_temperature": cross_entropy_at_temperature(logits, labels, 1.0),
            "nll_after_temperature": cross_entropy_at_temperature(logits, labels, temperature),
        }
    )
    return result


def calibrate_opinion_head(
    logits: Any,
    labels: Any,
    label2id: dict[str, int],
    target_precision: float,
) -> dict[str, Any]:
    torch = training.torch
    assert torch is not None
    if "Entirely factual" not in label2id:
        raise ValueError("Opinion labels must contain 'Entirely factual'.")
    factual_id = int(label2id["Entirely factual"])
    temperature = fit_temperature(logits, labels)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    predicted_ids = probabilities.argmax(dim=-1)
    scores = (1.0 - probabilities[:, factual_id]).tolist()
    truth = (labels != factual_id).tolist()
    candidates = (predicted_ids != factual_id).tolist()
    result = choose_threshold_for_precision(scores, truth, candidates, target_precision)
    result.update(
        {
            "temperature": temperature,
            "score": "1 - P(Entirely factual)",
            "positive_label": "not Entirely factual",
            "nll_before_temperature": cross_entropy_at_temperature(logits, labels, 1.0),
            "nll_after_temperature": cross_entropy_at_temperature(logits, labels, temperature),
        }
    )
    return result


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    args = parse_args()
    metadata = apply_metadata_defaults(args)
    if not 0.0 <= args.target_precision <= 1.0:
        raise SystemExit("--target-precision must be between 0 and 1.")
    if args.validation_size <= 0:
        raise SystemExit("Calibration requires a non-empty validation split.")

    training.import_training_dependencies()
    torch = training.torch
    assert torch is not None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Using device: %s", device)

    train_pool_df, _test_df, data_bias_label2id, data_opinion_label2id = training.load_dataset(args)
    _train_df, val_df = training.split_dataset(train_pool_df, args.validation_size, args.seed)
    if val_df.empty:
        raise SystemExit("Validation split is empty; increase --validation-size.")
    LOGGER.info("Calibration validation rows: %d", len(val_df))

    model, bias_head, opinion_head, bias_label2id, opinion_label2id = infer_bias_llm.load_sbert_and_heads(
        model_dir=args.sbert_model_dir,
        heads_path=args.classification_heads,
        device=device,
    )
    if data_bias_label2id != bias_label2id:
        raise SystemExit("Dataset bias label mapping does not match the checkpoint.")
    if data_opinion_label2id != opinion_label2id:
        raise SystemExit("Dataset opinion label mapping does not match the checkpoint.")

    bias_logits, opinion_logits, bias_labels, opinion_labels = collect_validation_logits(
        model=model,
        bias_head=bias_head,
        opinion_head=opinion_head,
        val_df=val_df,
        batch_size=args.batch_size,
        device=device,
    )
    bias_calibration = calibrate_bias_head(
        bias_logits, bias_labels, bias_label2id, args.target_precision
    )
    opinion_calibration = calibrate_opinion_head(
        opinion_logits, opinion_labels, opinion_label2id, args.target_precision
    )

    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_dir": str(args.sbert_model_dir),
        "classification_heads": str(args.classification_heads or (args.sbert_model_dir / "classification_heads.pt")),
        "target_precision": args.target_precision,
        "data": {
            "data_dir": str(args.data_dir),
            "data_glob": args.data_glob,
            "train_data_glob": args.train_data_glob,
            "test_data_glob": args.test_data_glob,
            "validation_size": args.validation_size,
            "seed": args.seed,
            "validation_rows": len(val_df),
            "metadata_seed": metadata.get("seed"),
        },
        "label_mappings": {
            "bias_label2id": bias_label2id,
            "opinion_label2id": opinion_label2id,
        },
        "heads": {
            "bias": bias_calibration,
            "opinion": opinion_calibration,
        },
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    LOGGER.info("Wrote calibration to %s", args.output_file)
    LOGGER.info("Bias calibration: %s", bias_calibration)
    LOGGER.info("Opinion calibration: %s", opinion_calibration)


if __name__ == "__main__":
    main()
