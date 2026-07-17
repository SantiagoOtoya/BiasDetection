from __future__ import annotations

import sys
import unittest
import types
from pathlib import Path
from unittest import mock

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_locked_test


class LockedTestCommandTests(unittest.TestCase):
    def test_parser_exposes_one_shot_inputs_outputs_and_promotion_flag(self) -> None:
        argv = [
            "evaluate_locked_test.py",
            "--sbert-model-dir", "model",
            "--calibration-file", "model/calibration.json",
            "--artifact-manifest", "model/artifact_manifest.json",
            "--data-dir", "BABE_HF",
            "--split-manifest", "artifacts/split.json",
            "--canonical-data-manifest", "artifacts/data.json",
            "--gate-config", "promotion_gates.json",
            "--output-file", "artifacts/evaluation.json",
            "--evaluation-lock", "artifacts/locked_once.json",
            "--device", "cuda",
            "--promote-if-gates-pass",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = evaluate_locked_test.parse_args()
        self.assertEqual(args.device, "cuda")
        self.assertTrue(args.promote_if_gates_pass)
        self.assertEqual(args.output_file, Path("artifacts/evaluation.json"))

    def test_partition_loader_requests_only_locked_test_rows(self) -> None:
        records = pd.DataFrame([{"record_id": "locked-1"}])
        locked = pd.DataFrame([{"record_id": "locked-1"}])
        canonical = {"canonical_data_manifest_sha256": "canonical-hash"}
        split = {"schema_version": "split_manifest/v1"}
        with mock.patch.object(
            evaluate_locked_test.training,
            "load_canonical_dataset",
            return_value=(
                records,
                canonical,
                {},
                {"Non-biased": 0, "Biased": 1},
                {"objective_style": 0, "opinionated_style": 1},
            ),
        ), mock.patch.object(
            evaluate_locked_test.data_pipeline,
            "load_split_manifest",
            return_value=split,
        ), mock.patch.object(
            evaluate_locked_test.data_pipeline, "validate_split_manifest"
        ), mock.patch.object(
            evaluate_locked_test.data_pipeline,
            "records_for_partition",
            return_value=locked,
        ) as partition_rows:
            observed, _, _, _ = evaluate_locked_test._load_locked_partition(
                types.SimpleNamespace(), Path("split.json")
            )
        partition_rows.assert_called_once_with(records, split, "locked_test")
        self.assertIs(observed, locked)


if __name__ == "__main__":
    unittest.main()
