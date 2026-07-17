from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import artifact_manifest
import finetune_all_mpnet_babe as training


class LegacyHierarchicalOpinionHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        training.torch = torch
        training.np = np
        training.pd = pd
        training.WeightedRandomSampler = torch.utils.data.WeightedRandomSampler

    @property
    def opinion_labels(self) -> dict[str, int]:
        return dict(training.CANONICAL_OPINION_LABEL2ID)

    def test_reconstructed_probabilities_sum_to_one(self) -> None:
        outputs = torch.tensor([[0.0, 0.0], [1.2, -0.7], [-1.1, 2.4]])
        probabilities = training.reconstruct_hierarchical_opinion_probabilities(outputs)

        self.assertTrue(torch.all(probabilities >= 0.0))
        self.assertTrue(
            torch.allclose(probabilities.sum(dim=-1), torch.ones(outputs.shape[0]))
        )
        self.assertAlmostEqual(float(probabilities[0, 0]), 0.5)
        self.assertAlmostEqual(float(probabilities[0, 1]), 0.25)
        self.assertAlmostEqual(float(probabilities[0, 2]), 0.25)

    def test_conditional_loss_ignores_factual_second_stage_logit(self) -> None:
        labels = torch.tensor([0, 1, 2])
        outputs = torch.tensor([[0.2, -12.0], [0.7, -0.4], [1.1, 0.8]])
        changed = outputs.clone()
        changed[0, 1] = 12.0

        base_loss, base_stage_one, base_conditional, counts = training.hierarchical_opinion_loss(
            outputs,
            labels,
            middle_class_weight=1.0,
            focal_gamma=0.0,
            opinion_class_weights=None,
        )
        changed_loss, changed_stage_one, changed_conditional, changed_counts = (
            training.hierarchical_opinion_loss(
                changed,
                labels,
                middle_class_weight=1.0,
                focal_gamma=0.0,
                opinion_class_weights=None,
            )
        )

        self.assertEqual(counts["conditional_opinion_examples"], 2)
        self.assertEqual(changed_counts["conditional_opinion_examples"], 2)
        self.assertTrue(torch.allclose(base_stage_one, changed_stage_one))
        self.assertTrue(torch.allclose(base_conditional, changed_conditional))
        self.assertTrue(torch.allclose(base_loss, changed_loss))

    def test_legacy_flat_payload_loads_without_head_type(self) -> None:
        legacy_head = training.build_mlp_head(4, 3, 3, 0.0)
        payload = {
            "bias_head_state_dict": training.build_mlp_head(4, 3, 2, 0.0).state_dict(),
            "opinion_head_state_dict": legacy_head.state_dict(),
            "embedding_dim": 4,
            "head_hidden_dim": 3,
            "dropout": 0.0,
            "bias_label2id": {"Non-biased": 0, "Biased": 1},
            "opinion_label2id": self.opinion_labels,
        }

        loaded, head_type = training.load_opinion_head_from_checkpoint(
            payload, torch.device("cpu")
        )
        sample = torch.tensor([[0.4, -0.3, 0.2, 0.1]])
        legacy_head.eval()
        loaded.eval()
        self.assertEqual(head_type, training.LEGACY_FLAT_HEAD_TYPE)
        self.assertTrue(torch.allclose(legacy_head(sample), loaded(sample)))

    def test_declared_hierarchy_with_flat_state_dict_raises(self) -> None:
        payload = {
            "head_type": training.HIERARCHICAL_ORDINAL_HEAD_TYPE,
            "opinion_head_state_dict": training.build_mlp_head(4, 3, 3, 0.0).state_dict(),
            "embedding_dim": 4,
            "head_hidden_dim": 3,
            "dropout": 0.0,
            "opinion_label2id": self.opinion_labels,
        }

        with self.assertRaises(training.CheckpointCompatibilityError):
            training.load_opinion_head_from_checkpoint(payload, torch.device("cpu"))

    def test_checkpoint_and_artifact_metadata_identify_hierarchy(self) -> None:
        bias_head = training.build_mlp_head(4, 3, 2, 0.0)
        opinion_head = training.build_opinion_head(
            embedding_dim=4,
            hidden_dim=3,
            dropout=0.0,
            opinion_label2id=self.opinion_labels,
            head_type=training.HIERARCHICAL_ORDINAL_HEAD_TYPE,
        )
        payload = training.build_classification_heads_checkpoint_payload(
            bias_head=bias_head,
            opinion_head=opinion_head,
            embedding_dim=4,
            hidden_dim=3,
            dropout=0.0,
            bias_label2id={"Non-biased": 0, "Biased": 1},
            opinion_label2id=self.opinion_labels,
            head_type=training.HIERARCHICAL_ORDINAL_HEAD_TYPE,
        )
        self.assertEqual(payload["checkpoint_schema_version"], "classification_heads/v2")
        self.assertTrue(payload["legacy_compatibility"])
        self.assertEqual(payload["head_type"], training.HIERARCHICAL_ORDINAL_HEAD_TYPE)
        self.assertEqual(
            payload["opinion_head_architecture"]["logit_names"],
            ["q_nonfactual_logit", "q_writer_given_nonfactual_logit"],
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "model.bin").write_bytes(b"model")
            (root / "heads.pt").write_bytes(b"heads")
            (root / "split.json").write_text("{}", encoding="utf-8")
            (root / "data.json").write_text("{}", encoding="utf-8")
            (root / "score.json").write_text("{}", encoding="utf-8")
            (root / "training.py").write_text("pass\n", encoding="utf-8")
            manifest = artifact_manifest.build_artifact_manifest(
                root,
                model_path=model_dir,
                heads_path=root / "heads.pt",
                split_manifest_path=root / "split.json",
                canonical_data_manifest_path=root / "data.json",
                evaluation_scorecard_path=root / "score.json",
                code_paths=[root / "training.py"],
                label_mappings={"opinion_label2id": self.opinion_labels},
                head_type=payload["head_type"],
                head_architecture=payload["opinion_head_architecture"],
                training_configuration={"checkpoint_selection": "multi_objective"},
                source_summary={},
            )
        self.assertEqual(
            manifest["core"]["head_type"], training.HIERARCHICAL_ORDINAL_HEAD_TYPE
        )

    def test_multi_objective_checkpoint_order_prioritizes_middle_recall_after_f1(self) -> None:
        baseline = {
            "opinion_macro_f1": 0.70,
            "opinion_middle_recall": 0.45,
            "bias_macro_f1": 0.80,
            "loss": 0.50,
        }
        better_middle = {**baseline, "opinion_middle_recall": 0.55}
        lower_opinion_f1 = {**better_middle, "opinion_macro_f1": 0.69}

        baseline_key = training.multi_objective_checkpoint_key(
            baseline,
            classifier_mode=training.LEGACY_HIERARCHICAL_CLASSIFIER_MODE,
        )
        self.assertGreater(
            training.multi_objective_checkpoint_key(
                better_middle,
                classifier_mode=training.LEGACY_HIERARCHICAL_CLASSIFIER_MODE,
            ),
            baseline_key,
        )
        self.assertLess(
            training.multi_objective_checkpoint_key(
                lower_opinion_f1,
                classifier_mode=training.LEGACY_HIERARCHICAL_CLASSIFIER_MODE,
            ),
            training.multi_objective_checkpoint_key(
                better_middle,
                classifier_mode=training.LEGACY_HIERARCHICAL_CLASSIFIER_MODE,
            ),
        )

    def test_balanced_sampler_preserves_unlabeled_rows_for_bias_supervision(self) -> None:
        sampler = training.build_opinion_balanced_sampler(
            pd.DataFrame(
                {
                    "opinion_label_id": pd.Series([0, 0, 1, 2, pd.NA], dtype="Int64")
                }
            ),
            types.SimpleNamespace(
                opinion_sampling="balanced",
                opinion_head_type=training.HIERARCHICAL_ORDINAL_HEAD_TYPE,
                seed=17,
            ),
        )

        self.assertIsNotNone(sampler)
        self.assertEqual(sampler.num_samples, 5)
        self.assertEqual(float(sampler.weights[-1]), 1.0)
        self.assertGreater(float(sampler.weights[2]), float(sampler.weights[0]))

    def test_hierarchical_diagnostics_are_oracle_gated_and_report_all_routes(self) -> None:
        outputs = torch.tensor(
            [
                [-5.0, 12.0],
                [5.0, -5.0],
                [5.0, 5.0],
                [5.0, -5.0],
                [5.0, 5.0],
                [-5.0, -12.0],
            ]
        )
        labels = torch.tensor([0, 1, 2, 1, 2, 0])

        diagnostics = training.hierarchical_opinion_evaluation_diagnostics(
            outputs, labels
        )
        self.assertEqual(diagnostics["threshold_source"], "default_0.5")
        stage_one = diagnostics["q_nonfactual"]
        stage_two = diagnostics["q_writer_given_nonfactual"]
        routed = diagnostics["threshold_routed_pipeline"]
        probabilities = diagnostics["final_three_class_probabilities"]
        self.assertEqual(stage_one["support"], 6)
        self.assertEqual(
            stage_one["true_class_counts"], {"factual": 2, "non_factual": 4}
        )
        self.assertEqual(stage_two["support"], 4)
        self.assertEqual(stage_two["true_class_counts"], {"middle": 2, "writer": 2})
        self.assertTrue(stage_two["oracle_gated_by_true_nonfactual"])
        self.assertAlmostEqual(stage_one["roc_auc"], 1.0)
        self.assertAlmostEqual(stage_one["pr_auc"], 1.0)
        self.assertAlmostEqual(stage_two["roc_auc"], 1.0)
        self.assertAlmostEqual(stage_two["pr_auc"], 1.0)
        self.assertEqual(routed["true_class_counts"], probabilities["true_class_counts"])
        self.assertEqual(routed["support"], probabilities["support"])
        self.assertEqual(probabilities["probability_reconstruction"]["factual"], "1 - q_nonfactual")

        changed_factual_stage_two = outputs.clone()
        changed_factual_stage_two[0, 1] = -12.0
        changed = training.hierarchical_opinion_evaluation_diagnostics(
            changed_factual_stage_two, labels
        )
        self.assertEqual(
            stage_two,
            changed["q_writer_given_nonfactual"],
            "Factual examples must not affect oracle-gated stage-two metrics.",
        )

    def test_hierarchical_diagnostics_handle_single_class_auc_and_custom_thresholds(self) -> None:
        outputs = torch.tensor([[-2.0, 0.0], [-1.0, 1.0]])
        labels = torch.tensor([0, 0])
        diagnostics = training.hierarchical_opinion_evaluation_diagnostics(
            outputs,
            labels,
            stage_thresholds={
                "q_nonfactual": 0.75,
                "q_writer_given_nonfactual": 0.25,
            },
        )
        self.assertEqual(diagnostics["threshold_source"], "validation_or_calibration")
        self.assertEqual(diagnostics["q_nonfactual"]["threshold"], 0.75)
        self.assertEqual(
            diagnostics["q_writer_given_nonfactual"]["threshold"], 0.25
        )
        self.assertIsNone(diagnostics["q_nonfactual"]["roc_auc"])
        self.assertIsNone(diagnostics["q_nonfactual"]["pr_auc"])
        self.assertEqual(diagnostics["q_writer_given_nonfactual"]["support"], 0)
        self.assertIsNone(diagnostics["q_writer_given_nonfactual"]["roc_auc"])
        self.assertIsNone(diagnostics["q_writer_given_nonfactual"]["pr_auc"])

    def test_evaluate_attaches_and_logs_hierarchical_diagnostics(self) -> None:
        batch = {
            "texts": ["a", "b", "c"],
            "bias_labels": torch.tensor([0, 1, 0]),
            "opinion_labels": torch.tensor([0, 1, 2]),
        }
        bias_logits = torch.tensor([[4.0, -4.0], [-4.0, 4.0], [4.0, -4.0]])
        opinion_outputs = torch.tensor([[-5.0, 5.0], [5.0, -5.0], [5.0, 5.0]])
        zero = torch.tensor(0.0)
        loss_components = {
            "opinion_nonfactual_loss": zero,
            "opinion_conditional_loss": zero,
            "conditional_opinion_examples": 2,
        }
        with mock.patch.object(
            training,
            "batch_loss",
            return_value=(
                zero,
                zero,
                zero,
                bias_logits,
                opinion_outputs,
                loss_components,
            ),
        ):
            with self.assertLogs("finetune_all_mpnet_babe", level="INFO") as logs:
                metrics = training.evaluate(
                    model=torch.nn.Identity(),
                    bias_head=torch.nn.Identity(),
                    opinion_head=torch.nn.Identity(),
                    dataloader=[batch],
                    bias_criterion=torch.nn.CrossEntropyLoss(),
                    opinion_criterion=torch.nn.BCEWithLogitsLoss(),
                    opinion_class_weights=None,
                    opinion_loss_alpha=0.3,
                    opinion_head_type=training.HIERARCHICAL_ORDINAL_HEAD_TYPE,
                    middle_class_weight=1.0,
                    opinion_focal_gamma=0.0,
                    device=torch.device("cpu"),
                    use_amp=False,
                    bias_num_labels=2,
                    opinion_num_labels=3,
                    evaluation_name="test evaluation",
                )

        diagnostics = metrics["hierarchical_opinion_diagnostics"]
        self.assertEqual(diagnostics["q_nonfactual"]["support"], 3)
        self.assertEqual(diagnostics["q_writer_given_nonfactual"]["support"], 2)
        output = "\n".join(logs.output)
        self.assertIn("legacy hierarchy style_macro_f1=", output)
        self.assertNotIn("final_three_class_probabilities diagnostics", output)


if __name__ == "__main__":
    unittest.main()
