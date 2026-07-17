from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_ROOT / "LLM-inference"
for import_path in (PROJECT_ROOT, LLM_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import evidence_retrieval
import finetune_all_mpnet_babe as training


def load_infer_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "infer_bias_llm_calibrated", LLM_DIR / "infer_bias_llm.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["infer_bias_llm_calibrated"] = module
    spec.loader.exec_module(module)
    return module


infer_bias_llm = load_infer_module()


def region(enabled: bool, threshold: float | None, metric: str) -> object:
    return infer_bias_llm.ThresholdRegionConfig(
        enabled=enabled, threshold=threshold, metric=metric, target=0.90,
        confidence=0.95, minimum_predicted_support=30,
        selected_support=30 if enabled else 0,
        observed_metric=1.0 if enabled else None,
        wilson_lower_bound=0.90 if enabled else 0.0,
        coverage=0.30 if enabled else 0.0,
        disabled_reason=None if enabled else "minimum_predicted_support_not_met",
    )


def decision_head(lower: object, upper: object) -> object:
    return infer_bias_llm.DecisionHeadConfig(
        calibrator=infer_bias_llm.ProbabilityCalibratorConfig(
            method="temperature_scaling", input_kind="binary_logit", temperature=1.0
        ),
        lower=lower,
        upper=upper,
    )


def calibrated_config(*, include_possible_bias: bool = False, disable_upper: bool = False) -> object:
    return infer_bias_llm.SelectionConfig(
        requested_mode="calibrated", effective_mode="calibrated",
        bias_decision=decision_head(
            region(True, 0.20, "negative_predictive_value"),
            region(not disable_upper, None if disable_upper else 0.80, "precision"),
        ),
        opinion_style_decision=decision_head(
            region(True, 0.25, "negative_predictive_value"),
            region(True, 0.75, "precision"),
        ),
        include_possible_bias=include_possible_bias,
    )


class CalibratedDecisionTests(unittest.TestCase):
    def test_all_bias_and_style_states(self) -> None:
        clear = infer_bias_llm.evaluate_calibrated_decisions(0.80, 0.75, calibrated_config())
        middle = infer_bias_llm.evaluate_calibrated_decisions(0.50, 0.50, calibrated_config())
        low = infer_bias_llm.evaluate_calibrated_decisions(0.20, 0.25, calibrated_config())
        self.assertEqual((clear.bias_assessment, clear.opinion_style), ("clear_bias", "opinionated_style"))
        self.assertEqual((middle.bias_assessment, middle.opinion_style), ("possible_bias", "uncertain"))
        self.assertEqual((low.bias_assessment, low.opinion_style), ("no_clear_bias", "objective_style"))
        self.assertTrue(clear.selected)
        self.assertFalse(middle.selected)
        self.assertFalse(low.selected)

    def test_disabled_confident_region_falls_back_to_possible_bias(self) -> None:
        outcome = infer_bias_llm.evaluate_calibrated_decisions(
            0.99, 0.50, calibrated_config(disable_upper=True)
        )
        self.assertEqual(outcome.bias_assessment, "possible_bias")
        self.assertFalse(outcome.selected)

    def test_disabled_style_region_falls_back_to_uncertain(self) -> None:
        config = calibrated_config()
        config = infer_bias_llm.SelectionConfig(
            requested_mode="calibrated", effective_mode="calibrated",
            bias_decision=config.bias_decision,
            opinion_style_decision=decision_head(
                region(True, 0.25, "negative_predictive_value"),
                region(False, None, "precision"),
            ),
        )
        outcome = infer_bias_llm.evaluate_calibrated_decisions(0.90, 0.99, config)
        self.assertEqual(outcome.opinion_style, "uncertain")

    def test_possible_bias_requires_explicit_inclusion(self) -> None:
        default = infer_bias_llm.evaluate_calibrated_decisions(0.50, 0.50, calibrated_config())
        included = infer_bias_llm.evaluate_calibrated_decisions(
            0.50, 0.50, calibrated_config(include_possible_bias=True)
        )
        self.assertFalse(default.selected)
        self.assertTrue(included.selected)

    def test_opinion_style_cannot_trigger_or_suppress_bias_selection(self) -> None:
        objective = infer_bias_llm.evaluate_calibrated_decisions(0.90, 0.01, calibrated_config())
        opinionated = infer_bias_llm.evaluate_calibrated_decisions(0.90, 0.99, calibrated_config())
        self.assertTrue(objective.selected)
        self.assertTrue(opinionated.selected)
        self.assertEqual(objective.bias_assessment, opinionated.bias_assessment)

    def test_production_classification_uses_scalar_logits_and_new_json(self) -> None:
        training.torch = torch

        class StaticHead(torch.nn.Module):
            def __init__(self, value: float, head_type: str) -> None:
                super().__init__()
                self.value = value
                self.head_type = head_type

            def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
                return torch.full((embeddings.shape[0], 1), self.value)

        with mock.patch.object(
            infer_bias_llm.training, "sentence_embeddings", return_value=torch.zeros((1, 2))
        ):
            predictions = infer_bias_llm.classify_sentences(
                ["A sentence."], object(),
                StaticHead(2.0, training.BINARY_BIAS_HEAD_TYPE),
                StaticHead(-2.0, training.BINARY_OPINION_STYLE_HEAD_TYPE),
                {"Non-biased": 0, "Biased": 1},
                {"objective_style": 0, "opinionated_style": 1},
                torch.device("cpu"), 1, calibrated_config(),
            )
        record = infer_bias_llm.prediction_to_json(predictions[0])
        self.assertEqual(record["bias_assessment"], "clear_bias")
        self.assertEqual(record["opinion_style"], "objective_style")
        self.assertNotIn("opinion_label", record)
        self.assertNotIn("evidence_status", record)
        self.assertNotIn("supported", repr(record))
        self.assertNotIn("contradicted", repr(record))
        self.assertAlmostEqual(record["p_bias"], torch.sigmoid(torch.tensor(2.0)).item())

    def test_inference_applies_temperature_and_sigmoid_exactly_once(self) -> None:
        training.torch = torch
        config = calibrated_config()
        bias_decision = infer_bias_llm.DecisionHeadConfig(
            calibrator=infer_bias_llm.ProbabilityCalibratorConfig(
                method="temperature_scaling", input_kind="binary_logit", temperature=2.0
            ),
            lower=config.bias_decision.lower,
            upper=config.bias_decision.upper,
        )
        config = infer_bias_llm.SelectionConfig(
            requested_mode="calibrated",
            effective_mode="calibrated",
            bias_decision=bias_decision,
            opinion_style_decision=config.opinion_style_decision,
        )

        class StaticHead(torch.nn.Module):
            def __init__(self, value: float, head_type: str) -> None:
                super().__init__()
                self.value = value
                self.head_type = head_type

            def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
                return torch.full((embeddings.shape[0], 1), self.value)

        with mock.patch.object(
            infer_bias_llm.training, "sentence_embeddings", return_value=torch.zeros((1, 2))
        ):
            prediction = infer_bias_llm.classify_sentences(
                ["A sentence."], object(),
                StaticHead(2.0, training.BINARY_BIAS_HEAD_TYPE),
                StaticHead(0.0, training.BINARY_OPINION_STYLE_HEAD_TYPE),
                {"Non-biased": 0, "Biased": 1},
                {"objective_style": 0, "opinionated_style": 1},
                torch.device("cpu"), 1, config,
            )[0]
        expected = torch.sigmoid(torch.tensor(1.0)).item()
        self.assertAlmostEqual(prediction.p_bias, expected)
        self.assertNotAlmostEqual(
            prediction.p_bias, torch.sigmoid(torch.tensor(expected)).item()
        )

    def test_target_only_input_is_encoded_once_for_both_heads(self) -> None:
        training.torch = torch
        inputs = infer_bias_llm.build_sentence_context_inputs(
            ["Before.", "Target sentence.", "After."], context_sentences=1
        )

        class RecordingHead(torch.nn.Module):
            head_type = training.BINARY_BIAS_HEAD_TYPE

            def __init__(self) -> None:
                super().__init__()
                self.seen = None

            def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
                self.seen = embeddings
                return torch.ones((embeddings.shape[0], 1))

        bias_head = RecordingHead()
        style_head = RecordingHead()
        style_head.head_type = training.BINARY_OPINION_STYLE_HEAD_TYPE
        embeddings = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        with mock.patch.object(
            infer_bias_llm.training, "sentence_embeddings", return_value=embeddings
        ) as encode:
            infer_bias_llm.classify_sentences(
                inputs, object(), bias_head, style_head,
                {"Non-biased": 0, "Biased": 1},
                {"objective_style": 0, "opinionated_style": 1},
                torch.device("cpu"), 8, calibrated_config(),
            )
        encode.assert_called_once()
        encoded_texts = encode.call_args.args[1]
        self.assertEqual(encoded_texts[1], "[TARGET]\nTarget sentence.")
        self.assertNotIn("Before.", encoded_texts[1])
        self.assertNotIn("After.", encoded_texts[1])
        self.assertIs(bias_head.seen, embeddings)
        self.assertIs(style_head.seen, embeddings)

    def test_opinion_only_candidate_does_not_retrieve(self) -> None:
        prediction = infer_bias_llm.SentencePrediction(
            index=0, text="An opinionated sentence.",
            bias_label="Non-biased", bias_probability=0.9,
            bias_probabilities={"Non-biased": 0.9, "Biased": 0.1},
            opinion_label="opinionated_style", opinion_probability=0.99,
            opinion_probabilities={"objective_style": 0.01, "opinionated_style": 0.99},
            selected=False, bias_assessment="no_clear_bias",
            opinion_style="opinionated_style", legacy_compatibility=False,
        )
        args = types.SimpleNamespace(
            batch_size=16, context_sentences=0, max_windows_per_article=0,
            sbert_model_dir="model", llm_model="llm", enable_evidence=True,
            evidence_mode="web", evidence_provider="brave", max_evidence_items=5,
            evidence_timeout_seconds=1, include_all_sentence_predictions=False,
        )
        with mock.patch.object(infer_bias_llm, "classify_sentences", return_value=[prediction]):
            with mock.patch.object(infer_bias_llm.evidence_retrieval, "retrieve_evidence") as retrieve:
                record = infer_bias_llm.analyze_article(
                    infer_bias_llm.ArticleRecord("a1", prediction.text, {}),
                    None, None, None, {}, {}, None, args, None,
                    evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY,
                    calibrated_config(),
                )
        self.assertEqual(record["selected_sentence_count"], 0)
        retrieve.assert_not_called()

    def test_explicit_argmax_legacy_behavior_remains_isolated(self) -> None:
        outcome = infer_bias_llm.evaluate_sentence_selection(
            "Biased", "Entirely factual",
            {"Non-biased": 0.49, "Biased": 0.51},
            {"Entirely factual": 0.99},
            infer_bias_llm.legacy_argmax_selection_config(),
        )
        self.assertTrue(outcome.selected)
        self.assertIsNone(outcome.bias_assessment)


if __name__ == "__main__":
    unittest.main()
