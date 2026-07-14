"""Deterministic evaluation, confidence bounds, and release-gate helpers."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence


EVALUATION_SCORECARD_SCHEMA_VERSION = "evaluation_scorecard/v1"
BINARY_EVALUATION_SCORECARD_SCHEMA_VERSION = "evaluation_scorecard/v2"
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_GATE_TARGETS = {
    "bias_precision_lcb": 0.90,
    "article_report_precision_lcb": 0.90,
    "opinion_macro_f1_lcb": 0.65,
    "middle_recall_lcb": 0.50,
    "locked_test_bias_positive_support": 500,
    "locked_test_middle_support": 300,
}
DEFAULT_BINARY_GATE_TARGETS = {
    "clear_bias_precision_lcb": 0.90,
    "opinion_style_macro_f1_bootstrap_lcb": 0.65,
    "objective_style_recall_lcb": 0.50,
    "opinionated_style_recall_lcb": 0.50,
    "minimum_evaluable_support": 30,
}


def _validate_labels(
    truth: Sequence[int], predictions: Sequence[int], labels: Sequence[int]
) -> list[int]:
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same length.")
    ordered = list(labels)
    if len(set(ordered)) != len(ordered):
        raise ValueError("labels must be unique.")
    allowed = set(ordered)
    unknown = set(truth).union(predictions).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown labels in predictions or truth: {sorted(unknown)}")
    return ordered


def confusion_matrix(
    truth: Sequence[int], predictions: Sequence[int], labels: Sequence[int]
) -> list[list[int]]:
    """Return rows=true / columns=predicted counts in the supplied label order."""

    ordered = _validate_labels(truth, predictions, labels)
    positions = {label: index for index, label in enumerate(ordered)}
    matrix = [[0 for _ in ordered] for _ in ordered]
    for actual, predicted in zip(truth, predictions):
        matrix[positions[actual]][positions[predicted]] += 1
    return matrix


def classification_metrics(
    truth: Sequence[int],
    predictions: Sequence[int],
    labels: Sequence[int],
    *,
    label_names: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Calculate count-aware per-class metrics and a stable confusion matrix."""

    ordered = _validate_labels(truth, predictions, labels)
    matrix = confusion_matrix(truth, predictions, ordered)
    per_class: list[dict[str, Any]] = []
    f1_values: list[float] = []
    for index, label in enumerate(ordered):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(ordered)) if row != index)
        false_negative = sum(matrix[index][column] for column in range(len(ordered)) if column != index)
        support = sum(matrix[index])
        predicted_support = sum(matrix[row][index] for row in range(len(ordered)))
        precision = true_positive / predicted_support if predicted_support else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append(
            {
                "label_id": int(label),
                "label": label_names.get(label, str(label)) if label_names else str(label),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "support": support,
                "predicted_support": predicted_support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    correct = sum(matrix[index][index] for index in range(len(ordered)))
    return {
        "labels": [int(label) for label in ordered],
        "label_names": [label_names.get(label, str(label)) if label_names else str(label) for label in ordered],
        "confusion_matrix": matrix,
        "accuracy": correct / len(truth) if truth else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
    }


def _binary_roc_auc(truth: Sequence[int], scores: Sequence[float]) -> float | None:
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


def _binary_average_precision(
    truth: Sequence[int], scores: Sequence[float]
) -> float | None:
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


def binary_classification_metrics(
    truth: Sequence[int],
    probabilities: Sequence[float],
    *,
    negative_label: str,
    positive_label: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return fixed-threshold binary diagnostics without tuning a threshold."""

    if len(truth) != len(probabilities):
        raise ValueError("truth and probabilities must have the same length.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one.")
    normalized_truth = [int(value) for value in truth]
    if not set(normalized_truth).issubset({0, 1}):
        raise ValueError("binary truth values must be 0 or 1.")
    normalized_probabilities = [float(value) for value in probabilities]
    if any(value < 0.0 or value > 1.0 for value in normalized_probabilities):
        raise ValueError("binary probabilities must be between zero and one.")
    predictions = [int(value >= threshold) for value in normalized_probabilities]
    result = classification_metrics(
        normalized_truth,
        predictions,
        labels=(0, 1),
        label_names={0: negative_label, 1: positive_label},
    )
    recalls = [
        float(item["recall"])
        for item in result["per_class"]
        if int(item["support"]) > 0
    ]
    result.update(
        {
            "support": len(normalized_truth),
            "threshold": float(threshold),
            "balanced_accuracy": (
                sum(recalls) / len(recalls) if recalls else 0.0
            ),
            "roc_auc": _binary_roc_auc(
                normalized_truth, normalized_probabilities
            ),
            "pr_auc": _binary_average_precision(
                normalized_truth, normalized_probabilities
            ),
            "pr_auc_definition": "average_precision",
        }
    )
    return result


def one_sided_wilson_lower_bound(
    successes: int,
    trials: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """One-sided Wilson lower confidence bound for a binomial proportion."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one.")
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes must satisfy 0 <= successes <= trials.")
    if trials == 0:
        return 0.0
    z_value = NormalDist().inv_cdf(confidence)
    proportion = successes / trials
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / trials
    center = proportion + z_squared / (2.0 * trials)
    adjustment = z_value * math.sqrt(
        (proportion * (1.0 - proportion) + z_squared / (4.0 * trials)) / trials
    )
    return max(0.0, (center - adjustment) / denominator)


def binary_decision_metrics(
    truth: Sequence[bool],
    selected: Sequence[bool],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    if len(truth) != len(selected):
        raise ValueError("truth and selected must have the same length.")
    true_positive = sum(bool(actual) and bool(predicted) for actual, predicted in zip(truth, selected))
    false_positive = sum(not bool(actual) and bool(predicted) for actual, predicted in zip(truth, selected))
    false_negative = sum(bool(actual) and not bool(predicted) for actual, predicted in zip(truth, selected))
    selected_count = true_positive + false_positive
    positive_support = true_positive + false_negative
    precision = true_positive / selected_count if selected_count else 0.0
    recall = true_positive / positive_support if positive_support else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "selected_count": selected_count,
        "positive_support": positive_support,
        "coverage": selected_count / len(truth) if truth else 0.0,
        "precision": precision,
        "recall": recall,
        "precision_wilson_lower_bound": one_sided_wilson_lower_bound(
            true_positive, selected_count, confidence
        ),
        "recall_wilson_lower_bound": one_sided_wilson_lower_bound(
            true_positive, positive_support, confidence
        ),
        "confidence": confidence,
    }


def binary_metric_wilson_bounds(
    metrics: Mapping[str, Any], *, confidence: float = DEFAULT_CONFIDENCE
) -> dict[str, Any]:
    """Attach one-sided Wilson bounds to every binary class precision/recall."""

    result = dict(metrics)
    classes: list[dict[str, Any]] = []
    for raw in metrics.get("per_class", []):
        item = dict(raw)
        item["precision_wilson_lower_bound"] = one_sided_wilson_lower_bound(
            int(item["true_positive"]), int(item["predicted_support"]), confidence
        )
        item["recall_wilson_lower_bound"] = one_sided_wilson_lower_bound(
            int(item["true_positive"]), int(item["support"]), confidence
        )
        classes.append(item)
    result["per_class"] = classes
    result["confidence"] = confidence
    return result


def decision_state_coverage(
    states: Sequence[str], *, ordered_states: Sequence[str]
) -> dict[str, Any]:
    allowed = set(ordered_states)
    unknown = set(states).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown decision states: {sorted(unknown)}")
    total = len(states)
    return {
        state: {
            "count": sum(value == state for value in states),
            "percentage": (sum(value == state for value in states) / total if total else 0.0),
        }
        for state in ordered_states
    }


def evaluate_binary_acceptance_gates(
    *,
    bias_truth: Sequence[int],
    bias_probabilities: Sequence[float],
    bias_states: Sequence[str],
    opinion_style_truth: Sequence[int],
    opinion_style_probabilities: Sequence[float],
    opinion_style_states: Sequence[str],
    group_ids: Sequence[str],
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
    targets: Mapping[str, float | int] | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen binary classifier without tuning any decision policy."""

    if not (len(bias_truth) == len(bias_probabilities) == len(bias_states)):
        raise ValueError("Bias locked-test arrays must have matching lengths.")
    if not (
        len(opinion_style_truth)
        == len(opinion_style_probabilities)
        == len(opinion_style_states)
        == len(group_ids)
    ):
        raise ValueError("Opinion-style locked-test arrays must have matching lengths.")
    merged_targets = dict(DEFAULT_BINARY_GATE_TARGETS)
    if targets:
        merged_targets.update(targets)
    bias_metrics = binary_metric_wilson_bounds(
        binary_classification_metrics(
            bias_truth, bias_probabilities,
            negative_label="not_biased", positive_label="biased",
        ),
        confidence=confidence,
    )
    style_metrics = binary_metric_wilson_bounds(
        binary_classification_metrics(
            opinion_style_truth, opinion_style_probabilities,
            negative_label="objective_style", positive_label="opinionated_style",
        ),
        confidence=confidence,
    )
    style_predictions = [int(float(value) >= 0.5) for value in opinion_style_probabilities]
    style_macro_lcb = article_stratified_bootstrap_lower_bound(
        opinion_style_truth, style_predictions, group_ids, labels=(0, 1),
        confidence=confidence, resamples=bootstrap_resamples, seed=seed,
    )
    clear_bias = binary_decision_metrics(
        [int(value) == 1 for value in bias_truth],
        [value == "clear_bias" for value in bias_states],
        confidence=confidence,
    )
    style_by_id = {int(item["label_id"]): item for item in style_metrics["per_class"]}
    minimum_support = int(merged_targets["minimum_evaluable_support"])
    failures: list[str] = []
    if clear_bias["selected_count"] < minimum_support:
        failures.append("insufficient_clear_bias_selected_support")
    if style_by_id[0]["support"] < minimum_support:
        failures.append("insufficient_objective_style_support")
    if style_by_id[1]["support"] < minimum_support:
        failures.append("insufficient_opinionated_style_support")
    if clear_bias["precision_wilson_lower_bound"] < float(
        merged_targets["clear_bias_precision_lcb"]
    ):
        failures.append("clear_bias_precision_lcb_below_target")
    if style_macro_lcb < float(
        merged_targets["opinion_style_macro_f1_bootstrap_lcb"]
    ):
        failures.append("opinion_style_macro_f1_bootstrap_lcb_below_target")
    if style_by_id[0]["recall_wilson_lower_bound"] < float(
        merged_targets["objective_style_recall_lcb"]
    ):
        failures.append("objective_style_recall_lcb_below_target")
    if style_by_id[1]["recall_wilson_lower_bound"] < float(
        merged_targets["opinionated_style_recall_lcb"]
    ):
        failures.append("opinionated_style_recall_lcb_below_target")
    return {
        "schema_version": BINARY_EVALUATION_SCORECARD_SCHEMA_VERSION,
        "confidence": confidence,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": seed,
        "targets": merged_targets,
        "bias": bias_metrics,
        "opinion_style": {
            **style_metrics,
            "macro_f1_bootstrap_lower_bound": style_macro_lcb,
        },
        "clear_bias": clear_bias,
        "decision_state_coverage": {
            "bias": decision_state_coverage(
                bias_states,
                ordered_states=("no_clear_bias", "possible_bias", "clear_bias"),
            ),
            "opinion_style": decision_state_coverage(
                opinion_style_states,
                ordered_states=("objective_style", "uncertain", "opinionated_style"),
            ),
        },
        "acceptance_gate": {
            "passed": not failures,
            "failures": failures,
            "promotion_state": "eligible" if not failures else "rejected",
        },
    }


def article_stratified_bootstrap_lower_bound(
    truth: Sequence[int],
    predictions: Sequence[int],
    group_ids: Sequence[str],
    labels: Sequence[int],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> float:
    """Deterministically bootstrap macro-F1 by article/event split group."""

    _validate_labels(truth, predictions, labels)
    if len(group_ids) != len(truth):
        raise ValueError("group_ids must have the same length as truth.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one.")
    if resamples <= 0:
        raise ValueError("resamples must be positive.")
    if not truth:
        return 0.0
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[str(group_id)].append(index)
    ordered_groups = sorted(groups)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample_truth: list[int] = []
        sample_predictions: list[int] = []
        for _ in ordered_groups:
            group_id = ordered_groups[rng.randrange(len(ordered_groups))]
            for index in groups[group_id]:
                sample_truth.append(int(truth[index]))
                sample_predictions.append(int(predictions[index]))
        values.append(classification_metrics(sample_truth, sample_predictions, labels)["macro_f1"])
    values.sort()
    lower_index = max(0, math.ceil((1.0 - confidence) * resamples) - 1)
    return float(values[lower_index])


def _middle_metrics(opinion_metrics: Mapping[str, Any], confidence: float) -> dict[str, Any]:
    for item in opinion_metrics.get("per_class", []):
        if item["label_id"] == 1:
            return {
                "support": item["support"],
                "true_positive": item["true_positive"],
                "recall": item["recall"],
                "recall_wilson_lower_bound": one_sided_wilson_lower_bound(
                    int(item["true_positive"]), int(item["support"]), confidence
                ),
            }
    return {
        "support": 0,
        "true_positive": 0,
        "recall": 0.0,
        "recall_wilson_lower_bound": 0.0,
    }


def _article_report_metrics(
    report_relevant: Sequence[bool | None] | None,
    report_selected: Sequence[bool] | None,
    group_ids: Sequence[str],
    confidence: float,
) -> dict[str, Any] | None:
    if report_relevant is None or report_selected is None:
        return None
    if not (len(report_relevant) == len(report_selected) == len(group_ids)):
        raise ValueError("Article report inputs must have matching lengths.")
    grouped: dict[str, dict[str, Any]] = {}
    for gold, selected, group_id in zip(report_relevant, report_selected, group_ids):
        if gold is None:
            continue
        entry = grouped.setdefault(str(group_id), {"gold": False, "selected": False})
        entry["gold"] = entry["gold"] or bool(gold)
        entry["selected"] = entry["selected"] or bool(selected)
    if not grouped:
        return None
    return binary_decision_metrics(
        [entry["gold"] for _, entry in sorted(grouped.items())],
        [entry["selected"] for _, entry in sorted(grouped.items())],
        confidence=confidence,
    )


def evaluate_acceptance_gates(
    *,
    bias_truth: Sequence[int],
    bias_selected: Sequence[bool],
    opinion_truth: Sequence[int],
    opinion_predictions: Sequence[int],
    group_ids: Sequence[str] | None = None,
    report_group_ids: Sequence[str] | None = None,
    report_relevant: Sequence[bool | None] | None = None,
    report_selected: Sequence[bool] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
    targets: Mapping[str, float | int] | None = None,
) -> dict[str, Any]:
    """Build a deterministic locked-test scorecard and promotion decision."""

    if len(bias_truth) != len(bias_selected):
        raise ValueError("bias_truth and bias_selected must have matching lengths.")
    if len(opinion_truth) != len(opinion_predictions):
        raise ValueError("opinion_truth and opinion_predictions must have matching lengths.")
    if group_ids is None:
        raise ValueError("group_ids are required for article-stratified opinion metrics.")
    if len(group_ids) != len(opinion_truth):
        raise ValueError("group_ids must match the opinion arrays.")
    effective_report_group_ids = (
        report_group_ids if report_group_ids is not None else group_ids
    )
    if report_relevant is not None or report_selected is not None:
        if report_relevant is None or report_selected is None:
            raise ValueError("report_relevant and report_selected must be provided together.")
        if not (
            len(report_relevant)
            == len(report_selected)
            == len(effective_report_group_ids)
        ):
            raise ValueError("Report arrays and report_group_ids must have matching lengths.")
    merged_targets = dict(DEFAULT_GATE_TARGETS)
    if targets:
        merged_targets.update(targets)
    bias_metrics = binary_decision_metrics(
        [int(label) == 1 for label in bias_truth], bias_selected, confidence=confidence
    )
    opinion_metrics = classification_metrics(
        opinion_truth,
        opinion_predictions,
        labels=(0, 1, 2),
        label_names={
            0: "Entirely factual",
            1: "Somewhat factual but also opinionated",
            2: "Expresses writer's opinion",
        },
    )
    macro_f1_lcb = article_stratified_bootstrap_lower_bound(
        opinion_truth,
        opinion_predictions,
        group_ids,
        labels=(0, 1, 2),
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    middle = _middle_metrics(opinion_metrics, confidence)
    article_report = _article_report_metrics(
        report_relevant, report_selected, effective_report_group_ids, confidence
    )
    failures: list[str] = []
    if bias_metrics["positive_support"] < int(merged_targets["locked_test_bias_positive_support"]):
        failures.append("insufficient_locked_test_bias_positive_support")
    if middle["support"] < int(merged_targets["locked_test_middle_support"]):
        failures.append("insufficient_locked_test_middle_support")
    if bias_metrics["precision_wilson_lower_bound"] < float(merged_targets["bias_precision_lcb"]):
        failures.append("bias_positive_precision_lcb_below_target")
    if article_report is None:
        failures.append("article_report_precision_gate_not_evaluable")
    elif article_report["precision_wilson_lower_bound"] < float(merged_targets["article_report_precision_lcb"]):
        failures.append("article_report_precision_lcb_below_target")
    if macro_f1_lcb < float(merged_targets["opinion_macro_f1_lcb"]):
        failures.append("opinion_macro_f1_lcb_below_target")
    if middle["recall_wilson_lower_bound"] < float(merged_targets["middle_recall_lcb"]):
        failures.append("middle_recall_lcb_below_target")
    return {
        "schema_version": EVALUATION_SCORECARD_SCHEMA_VERSION,
        "confidence": confidence,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": seed,
        "targets": merged_targets,
        "bias_positive": bias_metrics,
        "article_report": article_report,
        "opinion": {
            **opinion_metrics,
            "macro_f1_bootstrap_lower_bound": macro_f1_lcb,
            "middle_class": middle,
        },
        "acceptance_gate": {
            "passed": not failures,
            "failures": failures,
            "promotion_state": "eligible" if not failures else "rejected",
        },
    }
