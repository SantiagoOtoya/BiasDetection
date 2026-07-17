from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluation


class EvaluationConfidenceTests(unittest.TestCase):
    def test_binary_metrics_report_balanced_per_class_and_ranking_scores(self) -> None:
        metrics = evaluation.binary_classification_metrics(
            [0, 0, 1, 1],
            [0.1, 0.7, 0.8, 0.9],
            negative_label="objective_style",
            positive_label="opinionated_style",
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["pr_auc"], 1.0)
        self.assertEqual(
            [item["label"] for item in metrics["per_class"]],
            ["objective_style", "opinionated_style"],
        )

    def test_confusion_matrix_and_per_class_counts(self) -> None:
        metrics = evaluation.classification_metrics(
            truth=[0, 1, 2, 1],
            predictions=[0, 2, 2, 0],
            labels=[0, 1, 2],
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [1, 0, 1], [0, 0, 1]])
        middle = metrics["per_class"][1]
        self.assertEqual(middle["true_positive"], 0)
        self.assertEqual(middle["false_positive"], 0)
        self.assertEqual(middle["false_negative"], 2)
        self.assertEqual(middle["support"], 2)
        self.assertEqual(middle["recall"], 0.0)

    def test_one_sided_wilson_reference_and_boundaries(self) -> None:
        lower_bound = evaluation.one_sided_wilson_lower_bound(182, 199, 0.95)
        self.assertAlmostEqual(lower_bound, 0.876, places=3)
        self.assertEqual(evaluation.one_sided_wilson_lower_bound(0, 0), 0.0)
        with self.assertRaises(ValueError):
            evaluation.one_sided_wilson_lower_bound(2, 1)

    def test_bootstrap_is_deterministic(self) -> None:
        arguments = {
            "truth": [0, 1, 2, 0, 1, 2],
            "predictions": [0, 1, 2, 0, 2, 2],
            "group_ids": ["a", "a", "b", "c", "c", "d"],
            "labels": [0, 1, 2],
            "resamples": 80,
            "seed": 7,
        }
        self.assertEqual(
            evaluation.article_stratified_bootstrap_lower_bound(**arguments),
            evaluation.article_stratified_bootstrap_lower_bound(**arguments),
        )

    def test_acceptance_gate_reports_named_failures(self) -> None:
        result = evaluation.evaluate_acceptance_gates(
            bias_truth=[1, 0, 1],
            bias_selected=[True, True, False],
            opinion_truth=[0, 1, 2],
            opinion_predictions=[2, 2, 2],
            group_ids=["a", "b", "c"],
            report_relevant=[True, False, False],
            report_selected=[True, True, False],
            bootstrap_resamples=30,
            targets={
                "locked_test_bias_positive_support": 1,
                "locked_test_middle_support": 1,
            },
        )
        failures = result["acceptance_gate"]["failures"]
        self.assertIn("bias_positive_precision_lcb_below_target", failures)
        self.assertIn("article_report_precision_lcb_below_target", failures)
        self.assertIn("opinion_macro_f1_lcb_below_target", failures)
        self.assertIn("middle_recall_lcb_below_target", failures)
        self.assertFalse(result["acceptance_gate"]["passed"])

    def test_acceptance_gate_passes_with_sufficient_perfect_locked_test(self) -> None:
        bias_truth = [1] * 500 + [0] * 100
        bias_selected = [True] * 500 + [False] * 100
        opinion_truth = [1] * 300 + [0] * 150 + [2] * 150
        report_relevant = [True] * 500 + [False] * 100
        report_selected = [True] * 500 + [False] * 100
        result = evaluation.evaluate_acceptance_gates(
            bias_truth=bias_truth,
            bias_selected=bias_selected,
            opinion_truth=opinion_truth,
            opinion_predictions=opinion_truth,
            group_ids=[f"article-{index}" for index in range(600)],
            report_relevant=report_relevant,
            report_selected=report_selected,
            bootstrap_resamples=40,
            seed=3,
        )
        self.assertTrue(result["acceptance_gate"]["passed"])
        self.assertEqual(result["acceptance_gate"]["failures"], [])

    def test_acceptance_gate_allows_independently_masked_task_rows(self) -> None:
        result = evaluation.evaluate_acceptance_gates(
            bias_truth=[1, 0],
            bias_selected=[True, False],
            opinion_truth=[0, 1, 2],
            opinion_predictions=[0, 1, 2],
            group_ids=["opinion-a", "opinion-b", "opinion-c"],
            report_group_ids=["report-a", "report-b", "report-c", "report-d"],
            report_relevant=[True, False, False, False],
            report_selected=[True, False, False, False],
            bootstrap_resamples=20,
            targets={
                "locked_test_bias_positive_support": 1,
                "locked_test_middle_support": 1,
            },
        )

        self.assertEqual(result["bias_positive"]["positive_support"], 1)
        self.assertEqual(result["opinion"]["middle_class"]["support"], 1)
        self.assertEqual(result["article_report"]["selected_count"], 1)

    def test_binary_release_gate_reports_wilson_bounds_and_state_coverage(self) -> None:
        bias_truth = [1] * 40 + [0] * 40
        bias_scores = [0.95] * 40 + [0.05] * 40
        style_truth = [0] * 40 + [1] * 40
        style_scores = [0.05] * 40 + [0.95] * 40
        result = evaluation.evaluate_binary_acceptance_gates(
            bias_truth=bias_truth,
            bias_probabilities=bias_scores,
            bias_states=["clear_bias"] * 40 + ["no_clear_bias"] * 40,
            opinion_style_truth=style_truth,
            opinion_style_probabilities=style_scores,
            opinion_style_states=["objective_style"] * 40
            + ["opinionated_style"] * 40,
            group_ids=[f"article-{index}" for index in range(80)],
            bootstrap_resamples=30,
        )
        self.assertTrue(result["acceptance_gate"]["passed"])
        self.assertEqual(result["schema_version"], "evaluation_scorecard/v2")
        self.assertIn(
            "precision_wilson_lower_bound", result["bias"]["per_class"][1]
        )
        self.assertEqual(
            result["decision_state_coverage"]["bias"]["clear_bias"]["count"], 40
        )

    def test_binary_release_gate_allows_independent_task_masks(self) -> None:
        result = evaluation.evaluate_binary_acceptance_gates(
            bias_truth=[1] * 30 + [0] * 30,
            bias_probabilities=[0.99] * 30 + [0.01] * 30,
            bias_states=["clear_bias"] * 30 + ["no_clear_bias"] * 30,
            opinion_style_truth=[0] * 35 + [1] * 35,
            opinion_style_probabilities=[0.01] * 35 + [0.99] * 35,
            opinion_style_states=["objective_style"] * 35
            + ["opinionated_style"] * 35,
            group_ids=[f"style-{index}" for index in range(70)],
            bootstrap_resamples=20,
        )
        self.assertTrue(result["acceptance_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
