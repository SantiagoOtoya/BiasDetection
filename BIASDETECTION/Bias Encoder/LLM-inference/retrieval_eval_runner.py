#!/usr/bin/env python
"""Retrieval-only evaluation; never imports classifier evaluation or training."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import evidence_retrieval
import retrieval_store
import runtime_config


SCHEMA_VERSION = "retrieval_eval/v1"
PINNED_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
PINNED_RERANKER_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
RECALL_KS = (1, 3, 5, 10)


class RetrievalEvalError(RuntimeError):
    """The review gate, dataset schema, or benchmark execution failed."""


@dataclass(frozen=True)
class DatasetValidation:
    valid: bool
    review_complete: bool
    case_count: int
    answerable_count: int
    negative_count: int
    development_count: int
    held_out_count: int
    scenario_count: int
    errors: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def load_dataset(path: Path | str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalEvalError("Could not read retrieval evaluation dataset") from exc
    if not isinstance(value, dict):
        raise RetrievalEvalError("Retrieval evaluation dataset must be a JSON object")
    return value


def validate_dataset(dataset: Mapping[str, Any]) -> DatasetValidation:
    errors: list[str] = []
    if dataset.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    cases = dataset.get("cases")
    scenarios = dataset.get("multi_claim_scenarios")
    if not isinstance(cases, list):
        errors.append("cases must be an array")
        cases = []
    if not isinstance(scenarios, list):
        errors.append("multi_claim_scenarios must be an array")
        scenarios = []
    ids: set[str] = set()
    answerable = development = held_out = 0
    sources: dict[str, int] = {}
    review_complete = dataset.get("review_status") == "human_reviewed"
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            errors.append(f"cases[{index}] must be an object")
            continue
        for field in (
            "case_id", "split", "claim", "answerable", "source_id", "difficulty",
            "relevant_document_ids", "relevant_chunk_spans", "relevance_judgments", "review",
        ):
            if field not in case:
                errors.append(f"cases[{index}].{field} is required")
        case_id = str(case.get("case_id") or "")
        if case_id in ids:
            errors.append(f"case_id {case_id!r} is duplicated")
        ids.add(case_id)
        split = case.get("split")
        development += split == "development"
        held_out += split == "held_out"
        if split not in {"development", "held_out"}:
            errors.append(f"cases[{index}].split is invalid")
        answerable += bool(case.get("answerable"))
        source_id = str(case.get("source_id") or "")
        sources[source_id] = sources.get(source_id, 0) + 1
        relevant = case.get("relevant_document_ids")
        if bool(case.get("answerable")) and not relevant:
            errors.append(f"cases[{index}] is answerable without a relevant document")
        if not bool(case.get("answerable")) and relevant:
            errors.append(f"cases[{index}] is unanswerable but has relevant documents")
        review = case.get("review") or {}
        reviewed = (
            review.get("status") == "reviewed"
            and bool(review.get("reviewer"))
            and bool(review.get("reviewed_at"))
        )
        if not reviewed:
            review_complete = False
        else:
            judgments = case.get("relevance_judgments")
            spans = case.get("relevant_chunk_spans")
            if not isinstance(judgments, list) or len(judgments) < 10:
                errors.append(f"cases[{index}] needs reviewed top-10 relevance judgments")
            if bool(case.get("answerable")) and (not isinstance(spans, list) or not spans):
                errors.append(f"cases[{index}] needs reviewed relevant chunk spans")
            for span in spans if isinstance(spans, list) else ():
                if not isinstance(span, Mapping) or not {
                    "document_id", "chunk_id", "start_word", "end_word", "relevance"
                }.issubset(span):
                    errors.append(f"cases[{index}] contains an invalid relevant chunk span")
    scenario_development = scenario_held_out = 0
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            errors.append(f"multi_claim_scenarios[{index}] must be an object")
            continue
        split = scenario.get("split")
        scenario_development += split == "development"
        scenario_held_out += split == "held_out"
        claim_ids = scenario.get("claim_ids")
        if not isinstance(claim_ids, list) or len(claim_ids) < 2 or any(value not in ids for value in claim_ids):
            errors.append(f"multi_claim_scenarios[{index}].claim_ids is invalid")
        review = scenario.get("review") or {}
        if review.get("status") != "reviewed" or not review.get("reviewer") or not review.get("reviewed_at"):
            review_complete = False
    expected_sources = {
        "us_bls", "us_gao", "us_bea", "us_census", "us_cbo",
        "us_federal_reserve", "us_cdc", "us_sec",
    }
    if len(cases) != 120:
        errors.append("dataset must contain exactly 120 atomic cases")
    if answerable != 96:
        errors.append("dataset must contain exactly 96 answerable cases")
    if len(cases) - answerable != 24:
        errors.append("dataset must contain exactly 24 unanswerable cases")
    if development != 80 or held_out != 40:
        errors.append("dataset split must be 80 development and 40 held-out")
    if set(sources) != expected_sources or any(sources.get(source) != 15 for source in expected_sources):
        errors.append("cases must be evenly stratified with 15 cases per source")
    if len(scenarios) != 24 or scenario_development != 16 or scenario_held_out != 8:
        errors.append("multi-claim scenarios must split 16 development and 8 held-out")
    return DatasetValidation(
        valid=not errors,
        review_complete=review_complete,
        case_count=len(cases),
        answerable_count=answerable,
        negative_count=len(cases) - answerable,
        development_count=development,
        held_out_count=held_out,
        scenario_count=len(scenarios),
        errors=tuple(errors),
    )


def evaluate_rankings(
    dataset: Mapping[str, Any],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    split: str,
    threshold: float | None = None,
) -> dict[str, Any]:
    cases = [case for case in dataset.get("cases", ()) if case.get("split") == split]
    by_id = {str(case["case_id"]): case for case in cases}
    filtered: dict[str, list[Mapping[str, Any]]] = {}
    latencies: list[float] = []
    for case_id, case in by_id.items():
        values = [
            item for item in rankings.get(case_id, ())
            if threshold is None or item.get("score") is None or float(item["score"]) >= threshold
        ]
        filtered[case_id] = values
        if values and values[0].get("latency_seconds") is not None:
            latencies.append(float(values[0]["latency_seconds"]))
    answerable_cases = [case for case in cases if case.get("answerable")]
    recalls: dict[str, float] = {}
    for k in RECALL_KS:
        hits = sum(_has_relevant(case, filtered.get(str(case["case_id"]), ())[:k]) for case in answerable_cases)
        recalls[f"recall_at_{k}"] = hits / len(answerable_cases) if answerable_cases else 0.0

    judged = irrelevant = unjudged = 0
    for case in cases:
        judgments = {
            str(value.get("candidate_id")): str(value.get("relevance"))
            for value in case.get("relevance_judgments", ()) if isinstance(value, Mapping)
        }
        for result in filtered.get(str(case["case_id"]), ())[:5]:
            candidate_id = _candidate_id(result)
            judgment = judgments.get(candidate_id)
            if judgment is None:
                unjudged += 1
            else:
                judged += 1
                irrelevant += judgment == "irrelevant"
    coverage_values: list[float] = []
    for scenario in dataset.get("multi_claim_scenarios", ()):
        if scenario.get("split") != split:
            continue
        claim_ids = [value for value in scenario.get("claim_ids", ()) if value in by_id]
        covered = sum(
            _has_relevant(by_id[claim_id], filtered.get(claim_id, ())[:10])
            for claim_id in claim_ids
        )
        coverage_values.append(covered / len(claim_ids) if claim_ids else 0.0)
    return {
        "split": split,
        "threshold": threshold,
        **recalls,
        "irrelevant_result_rate_at_5": irrelevant / judged if judged else None,
        "judged_results_at_5": judged,
        "unjudged_results_at_5": unjudged,
        "multi_claim_coverage_at_10": statistics.mean(coverage_values) if coverage_values else 0.0,
        "latency": {
            "cold_seconds": latencies[0] if latencies else None,
            "warm_p50_seconds": _percentile(latencies[1:], 50),
            "warm_p95_seconds": _percentile(latencies[1:], 95),
        },
    }


def meets_gates(metrics: Mapping[str, Any], *, hybrid: bool = False) -> bool:
    irrelevant = metrics.get("irrelevant_result_rate_at_5")
    latency = (metrics.get("latency") or {}).get("warm_p95_seconds")
    return (
        float(metrics.get("recall_at_5") or 0.0) >= 0.80
        and irrelevant is not None
        and float(irrelevant) <= 0.30
        and float(metrics.get("multi_claim_coverage_at_10") or 0.0) >= 0.80
        and (not hybrid or (latency is not None and float(latency) <= 10.0))
    )


def select_threshold(
    dataset: Mapping[str, Any],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    hybrid: bool = False,
) -> tuple[float | None, dict[str, Any]]:
    baseline = evaluate_rankings(dataset, rankings, split="development", threshold=None)
    if meets_gates(baseline, hybrid=hybrid):
        return None, baseline
    scores = sorted({
        round(float(item["score"]), 6)
        for values in rankings.values() for item in values
        if item.get("score") is not None
    })
    candidates: list[tuple[float, dict[str, Any]]] = []
    for threshold in scores:
        metrics = evaluate_rankings(dataset, rankings, split="development", threshold=threshold)
        if meets_gates(metrics, hybrid=hybrid):
            candidates.append((threshold, metrics))
    if not candidates:
        return None, baseline
    return min(
        candidates,
        key=lambda value: (
            float(value[1].get("irrelevant_result_rate_at_5") or 0.0),
            -float(value[1].get("recall_at_5") or 0.0),
            value[0],
        ),
    )


def retrieve_rankings(
    dataset: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    policy: evidence_retrieval.TrustedSourcePolicy,
) -> dict[str, list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    for case in dataset.get("cases", ()):
        if case.get("split") != split:
            continue
        case_id = str(case["case_id"])
        request = evidence_retrieval.EvidenceRequest(
            mode="web" if mode == "fetched-web" else mode,
            claims=(evidence_retrieval.EvidenceClaim(
                claim_id=case_id,
                text=str(case["claim"]),
                context_text=str(case["claim"]),
            ),),
            max_items=20,
            timeout_seconds=10.0,
            request_id=f"retrieval-eval:{case_id}",
        )
        started = time.perf_counter()
        result = evidence_retrieval.retrieve_evidence_request(request, policy=policy)
        latency = time.perf_counter() - started
        items = result.items
        if mode == "fetched-web":
            items = [item for item in items if item.material_kind == "source_excerpt"]
        rankings[case_id] = [
            {
                "candidate_id": _item_candidate_id(item),
                "document_id": item.provenance.document_id if item.provenance else None,
                "chunk_id": item.provenance.chunk_id if item.provenance else None,
                "score": item.provenance.score if item.provenance else None,
                "text": item.snippet,
                "latency_seconds": latency,
            }
            for item in items
        ]
    return rankings


def run_evaluation(dataset: Mapping[str, Any], *, modes: Sequence[str]) -> dict[str, Any]:
    validation = validate_dataset(dataset)
    if not validation.valid:
        raise RetrievalEvalError("Dataset schema failed: " + "; ".join(validation.errors))
    if not validation.review_complete:
        raise RetrievalEvalError(
            "Dataset is pending human review; held-out evaluation is review-gated"
        )
    policy = evidence_retrieval.load_trusted_source_policy(None)
    reports: dict[str, Any] = {}
    for mode in modes:
        development_rankings = retrieve_rankings(
            dataset, mode=mode, split="development", policy=policy
        )
        threshold, development = select_threshold(
            dataset,
            development_rankings,
            hybrid=mode == "hybrid",
        )
        selected = "dense" if meets_gates(development, hybrid=mode == "hybrid") else "reranker_required"
        if selected == "reranker_required":
            # Do not silently load/train a reranker. The pinned model is used only
            # after a reviewed development run proves dense+threshold insufficient.
            development_rankings = rerank_top_20(dataset, development_rankings, split="development")
            development = evaluate_rankings(dataset, development_rankings, split="development")
            selected = "pinned_cross_encoder" if meets_gates(development, hybrid=mode == "hybrid") else "failed"
        held_out_rankings = retrieve_rankings(
            dataset, mode=mode, split="held_out", policy=policy
        )
        if selected == "pinned_cross_encoder":
            held_out_rankings = rerank_top_20(dataset, held_out_rankings, split="held_out")
        held_out = evaluate_rankings(
            dataset,
            held_out_rankings,
            split="held_out",
            threshold=threshold if selected == "dense" else None,
        )
        reports[mode] = {
            "selected_configuration": selected,
            "similarity_threshold": threshold if selected == "dense" else None,
            "reranker": PINNED_RERANKER if selected == "pinned_cross_encoder" else None,
            "reranker_revision": (
                PINNED_RERANKER_REVISION if selected == "pinned_cross_encoder" else None
            ),
            "development": development,
            "held_out": held_out,
            "held_out_gates_passed": meets_gates(held_out, hybrid=mode == "hybrid"),
        }
    return {"schema_version": "retrieval_eval_report/v1", "modes": reports}


def rerank_top_20(
    dataset: Mapping[str, Any],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    split: str,
) -> dict[str, list[dict[str, Any]]]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RetrievalEvalError("sentence-transformers CrossEncoder is unavailable") from exc
    model = CrossEncoder(PINNED_RERANKER, revision=PINNED_RERANKER_REVISION)
    claims = {
        str(case["case_id"]): str(case["claim"])
        for case in dataset.get("cases", ()) if case.get("split") == split
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for case_id, values in rankings.items():
        top = [dict(value) for value in values[:20]]
        pairs = [(claims[case_id], str(value.get("text") or "")) for value in top]
        scores = model.predict(pairs, show_progress_bar=False) if pairs else []
        for value, score in zip(top, scores, strict=True):
            value["rerank_score"] = float(score)
        output[case_id] = sorted(top, key=lambda item: -float(item.get("rerank_score") or 0.0))
    return output


def _has_relevant(case: Mapping[str, Any], values: Sequence[Mapping[str, Any]]) -> bool:
    documents = {str(value) for value in case.get("relevant_document_ids", ())}
    chunks = {
        str(value.get("chunk_id"))
        for value in case.get("relevant_chunk_spans", ()) if isinstance(value, Mapping)
    }
    return any(
        str(value.get("document_id")) in documents or str(value.get("chunk_id")) in chunks
        for value in values
    )


def _candidate_id(value: Mapping[str, Any]) -> str:
    return str(value.get("candidate_id") or value.get("chunk_id") or value.get("document_id") or "")


def _item_candidate_id(item: evidence_retrieval.EvidenceItem) -> str:
    if item.provenance and item.provenance.chunk_id:
        return item.provenance.chunk_id
    if item.provenance and item.provenance.document_id:
        return item.provenance.document_id
    return item.canonical_url or item.url


def _percentile(values: Sequence[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the retrieval-only trusted corpus benchmark")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("retrieval_eval") / "trusted_retrieval_v1.json",
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--mode",
        choices=("corpus", "fetched-web", "hybrid", "all"),
        default="all",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        runtime_config.load_explicit_env_file(args.env_file)
        dataset = load_dataset(args.dataset)
        validation = validate_dataset(dataset)
        if args.validate_only:
            report: dict[str, Any] = {"status": "valid" if validation.valid else "invalid", **validation.to_json()}
        else:
            modes = ("corpus", "fetched-web", "hybrid") if args.mode == "all" else (args.mode,)
            report = run_evaluation(dataset, modes=modes)
            report["status"] = "passed" if all(
                value["held_out_gates_passed"] for value in report["modes"].values()
            ) else "failed"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 0 if report.get("status") in {"valid", "passed"} else 3
    except (RetrievalEvalError, runtime_config.RuntimeConfigurationError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
