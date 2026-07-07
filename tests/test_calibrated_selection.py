from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_ROOT / "LLM-inference"
for import_path in (PROJECT_ROOT, LLM_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import calibrate_sbert_heads
import evidence_retrieval


def load_infer_module() -> types.ModuleType:
    module_path = LLM_DIR / "infer_bias_llm.py"
    spec = importlib.util.spec_from_file_location("infer_bias_llm_calibrated", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load infer_bias_llm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["infer_bias_llm_calibrated"] = module
    spec.loader.exec_module(module)
    return module


infer_bias_llm = load_infer_module()


class ThresholdSelectionTests(unittest.TestCase):
    def test_choose_threshold_maximizes_recall_at_target_precision(self) -> None:
        result = calibrate_sbert_heads.choose_threshold_for_precision(
            scores=[0.95, 0.90, 0.80, 0.70],
            labels=[True, False, True, True],
            candidate_mask=[True, True, True, True],
            target_precision=0.75,
        )

        self.assertTrue(result["enabled"])
        self.assertEqual(result["threshold"], 0.70)
        self.assertEqual(result["validation_precision"], 0.75)
        self.assertEqual(result["validation_recall"], 1.0)

    def test_choose_threshold_disables_when_target_precision_unmet(self) -> None:
        result = calibrate_sbert_heads.choose_threshold_for_precision(
            scores=[0.95, 0.90, 0.10],
            labels=[False, False, True],
            candidate_mask=[True, True, False],
            target_precision=0.90,
        )

        self.assertFalse(result["enabled"])
        self.assertIsNone(result["threshold"])
        self.assertEqual(result["disabled_reason"], "target_precision_not_met")


class CalibratedSelectionTests(unittest.TestCase):
    def calibrated_config(self) -> object:
        return infer_bias_llm.SelectionConfig(
            requested_mode="calibrated",
            effective_mode="calibrated",
            bias=infer_bias_llm.CalibrationHeadConfig(
                enabled=True,
                threshold=0.80,
                temperature=1.0,
            ),
            opinion=infer_bias_llm.CalibrationHeadConfig(
                enabled=True,
                threshold=0.70,
                temperature=1.0,
            ),
        )

    def test_calibrated_bias_below_threshold_abstains(self) -> None:
        outcome = infer_bias_llm.evaluate_sentence_selection(
            bias_label="Biased",
            opinion_label="Entirely factual",
            bias_probabilities={"Non-biased": 0.21, "Biased": 0.79},
            opinion_probabilities={"Entirely factual": 0.90},
            selection_config=self.calibrated_config(),
        )

        self.assertFalse(outcome.selected)
        self.assertEqual(outcome.abstention_reasons, ["bias_below_threshold"])

    def test_calibrated_bias_threshold_met_selects(self) -> None:
        outcome = infer_bias_llm.evaluate_sentence_selection(
            bias_label="Biased",
            opinion_label="Entirely factual",
            bias_probabilities={"Non-biased": 0.19, "Biased": 0.81},
            opinion_probabilities={"Entirely factual": 0.90},
            selection_config=self.calibrated_config(),
        )

        self.assertTrue(outcome.selected)
        self.assertEqual(outcome.selection_reasons, ["bias_threshold_met"])

    def test_calibrated_opinion_threshold_uses_non_factual_probability(self) -> None:
        outcome = infer_bias_llm.evaluate_sentence_selection(
            bias_label="Non-biased",
            opinion_label="Somewhat factual but also opinionated",
            bias_probabilities={"Non-biased": 0.80, "Biased": 0.20},
            opinion_probabilities={
                "Entirely factual": 0.25,
                "Somewhat factual but also opinionated": 0.45,
                "Expresses writer's opinion": 0.30,
            },
            selection_config=self.calibrated_config(),
        )

        self.assertTrue(outcome.selected)
        self.assertEqual(outcome.selection_reasons, ["opinion_threshold_met"])
        self.assertAlmostEqual(outcome.opinion_selection_score or 0.0, 0.75)

    def test_argmax_selection_preserves_old_positive_rule(self) -> None:
        outcome = infer_bias_llm.evaluate_sentence_selection(
            bias_label="Biased",
            opinion_label="Entirely factual",
            bias_probabilities={"Non-biased": 0.49, "Biased": 0.51},
            opinion_probabilities={"Entirely factual": 0.99},
            selection_config=infer_bias_llm.default_selection_config(),
        )

        self.assertTrue(outcome.selected)
        self.assertEqual(outcome.selection_reasons, ["bias_argmax"])

    def test_abstained_article_does_not_retrieve_evidence(self) -> None:
        prediction = infer_bias_llm.SentencePrediction(
            index=0,
            text="The sentence is borderline.",
            bias_label="Biased",
            bias_probability=0.79,
            bias_probabilities={"Non-biased": 0.21, "Biased": 0.79},
            opinion_label="Entirely factual",
            opinion_probability=0.90,
            opinion_probabilities={"Entirely factual": 0.90},
            selected=False,
            abstention_reasons=["bias_below_threshold"],
        )
        args = types.SimpleNamespace(
            batch_size=16,
            context_sentences=0,
            max_windows_per_article=0,
            sbert_model_dir="model",
            llm_model="llm",
            enable_evidence=True,
            evidence_provider="brave",
            max_evidence_items=5,
            evidence_timeout_seconds=1,
            include_all_sentence_predictions=False,
        )

        with mock.patch.object(infer_bias_llm, "classify_sentences", return_value=[prediction]):
            with mock.patch.object(infer_bias_llm.evidence_retrieval, "retrieve_evidence") as retrieve:
                record = infer_bias_llm.analyze_article(
                    article=infer_bias_llm.ArticleRecord(
                        article_id="a1",
                        text=prediction.text,
                        metadata={},
                    ),
                    model=None,
                    bias_head=None,
                    opinion_head=None,
                    bias_label2id={},
                    opinion_label2id={},
                    device=None,
                    args=args,
                    llm=None,
                    evidence_policy=evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY,
                    selection_config=self.calibrated_config(),
                )

        self.assertEqual(record["context_windows"], [])
        self.assertEqual(record["selected_sentence_count"], 0)
        self.assertEqual(record["abstention_summary"]["candidate_abstained_count"], 1)
        retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
