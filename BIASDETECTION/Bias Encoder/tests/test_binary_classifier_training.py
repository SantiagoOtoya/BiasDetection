from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import finetune_all_mpnet_babe as training


class _RecordingHead(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.received = None

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        self.received = embeddings
        return embeddings[:, :1] * self.scale


class _SavableModel:
    def save(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "dummy_model.txt").write_text("model\n", encoding="utf-8")


class BinaryClassifierTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        training.torch = torch
        training.np = np
        training.pd = pd

    def test_default_mode_is_production_and_old_flag_is_explicit_legacy_alias(self) -> None:
        with mock.patch.object(sys, "argv", ["finetune_all_mpnet_babe.py"]):
            args = training.parse_args()
        self.assertEqual(args.classifier_mode, training.PRODUCTION_CLASSIFIER_MODE)
        self.assertIsNone(args.opinion_head_type)

        with mock.patch.object(
            sys,
            "argv",
            [
                "finetune_all_mpnet_babe.py",
                "--opinion-head-type",
                training.HIERARCHICAL_ORDINAL_HEAD_TYPE,
            ],
        ):
            legacy = training.parse_args()
        self.assertEqual(
            legacy.classifier_mode,
            training.LEGACY_HIERARCHICAL_CLASSIFIER_MODE,
        )

    def test_one_embedding_feeds_two_independent_scalar_heads(self) -> None:
        embeddings = torch.tensor(
            [[1.0, 0.5], [-1.0, 0.25], [0.3, -0.4]], requires_grad=True
        )
        bias_head = _RecordingHead(1.0)
        style_head = _RecordingHead(-0.5)
        batch = {
            "texts": ["a", "b", "c"],
            "bias_labels": torch.tensor([0, 1, 1]),
            "opinion_style_labels": torch.tensor(
                [training.LABEL_IGNORE_INDEX, 0, 1]
            ),
        }
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        with mock.patch.object(
            training, "sentence_embeddings", return_value=embeddings
        ) as encode:
            result = training.production_batch_loss(
                model=torch.nn.Identity(),
                bias_head=bias_head,
                opinion_style_head=style_head,
                batch=batch,
                bias_criterion=criterion,
                opinion_style_criterion=criterion,
                bias_class_weights=None,
                opinion_style_class_weights=None,
                opinion_style_loss_weight=0.3,
                device=torch.device("cpu"),
            )

        encode.assert_called_once()
        self.assertIs(bias_head.received, embeddings)
        self.assertIs(style_head.received, embeddings)
        self.assertEqual(tuple(result[3].shape), (3,))
        self.assertEqual(tuple(result[4].shape), (3,))
        self.assertEqual(result[5]["bias_supervised_examples"], 3)
        self.assertEqual(result[5]["opinion_style_supervised_examples"], 2)

        result[0].backward()
        self.assertIsNotNone(bias_head.scale.grad)
        self.assertIsNotNone(style_head.scale.grad)

    def test_production_checkpoint_has_only_accurate_binary_contract(self) -> None:
        bias_head = training.build_binary_head(
            4, 3, 0.0, head_type=training.BINARY_BIAS_HEAD_TYPE
        )
        style_head = training.build_binary_head(
            4, 3, 0.0, head_type=training.BINARY_OPINION_STYLE_HEAD_TYPE
        )
        payload = training.build_production_classification_heads_checkpoint_payload(
            bias_head=bias_head,
            opinion_style_head=style_head,
            embedding_dim=4,
            hidden_dim=3,
            dropout=0.0,
            bias_label2id={"Non-biased": 0, "Biased": 1},
            opinion_style_label2id=dict(training.OPINION_STYLE_LABEL2ID),
        )

        self.assertEqual(payload["checkpoint_schema_version"], "classification_heads/v3")
        self.assertEqual(
            payload["classifier_input_contract"],
            training.data_pipeline.CLASSIFIER_INPUT_CONTRACT,
        )
        self.assertEqual(payload["bias_head_type"], training.BINARY_BIAS_HEAD_TYPE)
        self.assertEqual(
            payload["opinion_style_head_type"],
            training.BINARY_OPINION_STYLE_HEAD_TYPE,
        )
        self.assertNotIn("opinion_head_state_dict", payload)
        self.assertNotIn("opinion_label2id", payload)
        self.assertNotIn("q_nonfactual", repr(payload))
        self.assertNotIn("q_writer_given_nonfactual", repr(payload))

        loaded_bias, loaded_style, mode = (
            training.load_classification_heads_from_checkpoint(
                payload, torch.device("cpu")
            )
        )
        self.assertEqual(mode, training.PRODUCTION_CLASSIFIER_MODE)
        self.assertEqual(loaded_bias(torch.ones(2, 4)).shape, (2, 1))
        self.assertEqual(loaded_style(torch.ones(2, 4)).shape, (2, 1))

    def test_legacy_checkpoint_requires_explicit_loader_mode(self) -> None:
        legacy_head = training.build_mlp_head(4, 3, 3, 0.0)
        payload = {
            "bias_head_state_dict": training.build_mlp_head(4, 3, 2, 0.0).state_dict(),
            "opinion_head_state_dict": legacy_head.state_dict(),
            "embedding_dim": 4,
            "head_hidden_dim": 3,
            "dropout": 0.0,
            "bias_label2id": {"Non-biased": 0, "Biased": 1},
            "opinion_label2id": dict(training.CANONICAL_OPINION_LABEL2ID),
        }
        with self.assertRaises(training.CheckpointCompatibilityError):
            training.load_classification_heads_from_checkpoint(
                payload, torch.device("cpu")
            )
        _bias, _opinion, mode = training.load_classification_heads_from_checkpoint(
            payload,
            torch.device("cpu"),
            compatibility_mode="legacy",
        )
        self.assertEqual(mode, training.LEGACY_FLAT_CLASSIFIER_MODE)

    def test_production_outputs_use_formatted_json_and_style_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "model"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "finetune_all_mpnet_babe.py",
                    "--output-dir",
                    str(output),
                ],
            ):
                args = training.parse_args()
            bias_head = training.build_binary_head(
                4, 3, 0.0, head_type=training.BINARY_BIAS_HEAD_TYPE
            )
            style_head = training.build_binary_head(
                4, 3, 0.0, head_type=training.BINARY_OPINION_STYLE_HEAD_TYPE
            )
            empty = pd.DataFrame()
            training.save_outputs(
                output_dir=output,
                model=_SavableModel(),
                bias_head=bias_head,
                opinion_head=style_head,
                args=args,
                bias_label2id={"Non-biased": 0, "Biased": 1},
                opinion_label2id=dict(training.OPINION_STYLE_LABEL2ID),
                train_df=empty,
                val_df=empty,
                test_df=empty,
                parameter_summary={"total": 1, "trainable": 1, "frozen": 0},
                embedding_dim=4,
                training_history=[{"epoch": 1, "train": {"loss": 0.5}}],
                class_weight_summary={"bias": None, "opinion_style": None},
                final_test_metrics=None,
                best_validation_summary=None,
                foundation_context=None,
            )

            self.assertTrue((output / "opinion_style_label_mapping.json").exists())
            self.assertFalse((output / "opinion_label_mapping.json").exists())
            for name in (
                "bias_label_mapping.json",
                "opinion_style_label_mapping.json",
                "training_metadata.json",
                "training_history.json",
            ):
                text = (output / name).read_text(encoding="utf-8")
                self.assertTrue(text.endswith("\n"), name)
                self.assertIn("\n  ", text, name)

    def test_production_checkpoint_key_uses_binary_bias_and_style_metrics(self) -> None:
        baseline = {
            "bias_macro_f1": 0.80,
            "bias_biased_precision": 0.81,
            "bias_biased_recall": 0.79,
            "opinion_style_macro_f1": 0.74,
            "opinion_style_objective_recall": 0.72,
            "opinion_style_opinionated_recall": 0.76,
            "loss": 0.5,
        }
        better_precision = {**baseline, "bias_biased_precision": 0.82}
        better_style_but_worse_bias = {
            **baseline,
            "bias_macro_f1": 0.79,
            "opinion_style_macro_f1": 0.90,
        }
        baseline_key = training.multi_objective_checkpoint_key(baseline)
        self.assertGreater(
            training.multi_objective_checkpoint_key(better_precision), baseline_key
        )
        self.assertLess(
            training.multi_objective_checkpoint_key(better_style_but_worse_bias),
            baseline_key,
        )


if __name__ == "__main__":
    unittest.main()
