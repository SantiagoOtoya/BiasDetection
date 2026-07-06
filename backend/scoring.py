"""Bias intensity scoring.

Aggregates the SBERT per-sentence bias/opinion probabilities into a single
0-100 article score and a Low / Moderate / High / Severe bucket.

The score blends two signals so it reflects both how *pervasive* and how
*intense* the bias is:

  - coverage intensity: the mean bias signal across all sentences (pervasiveness)
  - peak intensity:     the mean bias signal of the most biased sentences (severity)

Each sentence's signal combines its probability of being ``Biased`` with the
probability mass it places on non-factual (opinionated) classes.
"""

from __future__ import annotations

from schemas import BiasLevel

BIAS_POSITIVE_LABEL = "Biased"
OPINION_FACTUAL_LABEL = "Entirely factual"

# Weighting between the bias head and the opinion head for a sentence's signal.
BIAS_WEIGHT = 0.65
OPINION_WEIGHT = 0.35

# Weighting between pervasiveness and severity for the article score.
COVERAGE_WEIGHT = 0.5
PEAK_WEIGHT = 0.5

# Fraction of sentences treated as the "peak" for the severity term.
PEAK_FRACTION = 0.15


def sentence_signal(bias_probabilities: dict[str, float], opinion_probabilities: dict[str, float]) -> float:
    """Per-sentence bias intensity in [0, 1]."""
    biased = float(bias_probabilities.get(BIAS_POSITIVE_LABEL, 0.0))
    factual = float(opinion_probabilities.get(OPINION_FACTUAL_LABEL, 0.0))
    non_factual = max(0.0, 1.0 - factual)
    signal = BIAS_WEIGHT * biased + OPINION_WEIGHT * non_factual
    return min(1.0, max(0.0, signal))


def compute_score(signals: list[float]) -> float:
    """Blend coverage and peak intensity into a 0-100 score."""
    if not signals:
        return 0.0

    coverage = sum(signals) / len(signals)

    ordered = sorted(signals, reverse=True)
    peak_count = max(1, round(len(ordered) * PEAK_FRACTION))
    peak = sum(ordered[:peak_count]) / peak_count

    blended = COVERAGE_WEIGHT * coverage + PEAK_WEIGHT * peak
    return round(min(100.0, max(0.0, blended * 100.0)), 1)


def bucket(score: float) -> BiasLevel:
    if score < 20:
        return BiasLevel.low
    if score < 45:
        return BiasLevel.moderate
    if score < 70:
        return BiasLevel.high
    return BiasLevel.severe


def caption(score: float, selected_count: int, sentence_count: int) -> str:
    level = bucket(score).value.lower()
    return (
        f"{level.capitalize()} bias — {selected_count} of {sentence_count} "
        f"sentence(s) flagged for review."
    )
