from __future__ import annotations

import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd
import torch
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_ROOT / "LLM-inference"
for import_path in (PROJECT_ROOT, LLM_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import calibrate_sbert_heads
import data_pipeline


class CalibrationLcbTests(unittest.TestCase):
    def test_upper_region_uses_wilson_lcb_not_point_precision(self) -> None:
        result = calibrate_sbert_heads.select_confident_region(
            [0.95, 0.90], [1, 1], direction="upper", target=0.90,
            minimum_support=1, confidence=0.95,
        )
        self.assertFalse(result["enabled"])
        self.assertEqual(result["observed_metric"], 1.0)
        self.assertLess(result["wilson_lower_bound"], 0.90)
        self.assertEqual(result["disabled_reason"], "target_wilson_lcb_not_met")

    def test_lower_region_uses_negative_predictive_value(self) -> None:
        result = calibrate_sbert_heads.select_confident_region(
            [0.05] * 30 + [0.90] * 10,
            [0] * 30 + [1] * 10,
            direction="lower", target=0.85, minimum_support=30,
        )
        self.assertTrue(result["enabled"])
        self.assertEqual(result["threshold"], 0.05)
        self.assertEqual(result["metric"], "negative_predictive_value")
        self.assertEqual(result["selected_support"], 30)

    def test_minimum_predicted_support_disables_region(self) -> None:
        result = calibrate_sbert_heads.select_confident_region(
            [0.9] * 20, [1] * 20, direction="upper", target=0.50,
            minimum_support=30,
        )
        self.assertFalse(result["enabled"])
        self.assertEqual(result["disabled_reason"], "minimum_predicted_support_not_met")
        self.assertEqual(result["selected_support"], 20)

    def test_threshold_search_is_deterministic_and_uses_boundary_tie_break(self) -> None:
        inputs = dict(
            scores=[0.99] * 30 + [0.80] * 30 + [0.10] * 10,
            labels=[1] * 60 + [0] * 10,
            direction="upper", target=0.85, minimum_support=30,
        )
        first = calibrate_sbert_heads.select_confident_region(**inputs)
        second = calibrate_sbert_heads.select_confident_region(**inputs)
        self.assertEqual(first, second)
        self.assertEqual(first["threshold"], 0.80)

    def test_binary_calibration_applies_sigmoid_once(self) -> None:
        previous = calibrate_sbert_heads.training.torch
        calibrate_sbert_heads.training.torch = torch
        try:
            calibrator = {
                "method": "temperature_scaling",
                "input_kind": "binary_logit",
                "parameters": {"temperature": 2.0},
            }
            observed = calibrate_sbert_heads.apply_binary_calibrator(
                torch.tensor([2.0]), calibrator
            )
            expected = torch.sigmoid(torch.tensor([1.0]))
        finally:
            calibrate_sbert_heads.training.torch = previous
        self.assertTrue(torch.allclose(observed, expected))
        self.assertFalse(torch.allclose(observed, torch.sigmoid(expected)))

    def test_binary_diagnostics_include_raw_and_calibrated_metrics(self) -> None:
        previous = calibrate_sbert_heads.training.torch
        calibrate_sbert_heads.training.torch = torch
        calibrator = {
            "method": "temperature_scaling",
            "input_kind": "binary_logit",
            "parameters": {"temperature": 1.0},
        }
        try:
            diagnostics = calibrate_sbert_heads.binary_calibration_diagnostics(
                torch.tensor([-2.0, 2.0]), torch.tensor([0, 1]), calibrator, bins=2
            )
        finally:
            calibrate_sbert_heads.training.torch = previous
        self.assertEqual(diagnostics["supervised_rows"], 2)
        self.assertIn("raw", diagnostics)
        self.assertIn("calibrated", diagnostics)

    def test_manifest_rejects_calibration_locked_test_group_reuse(self) -> None:
        source = pd.DataFrame([
            {
                "uuid": str(index), "text": f"Sentence {index}",
                "news_link": f"https://example.org/{index}",
                "label_bias": "Biased" if index % 2 else "Non-biased",
                "label_opinion": "Entirely factual",
            }
            for index in range(16)
        ])
        records = data_pipeline.canonicalize_records(source)
        records, stats = data_pipeline.clean_and_dedupe_records(records)
        data_manifest = data_pipeline.build_canonical_data_manifest(records, stats)
        _assigned, manifest = data_pipeline.build_group_split_manifest(
            records, seed=11,
            canonical_data_manifest_sha256=data_manifest["canonical_data_manifest_sha256"],
            cleaning_stats=stats, created_at_utc="2026-07-10T00:00:00Z",
        )
        calibration = next(x for x in manifest["assignments"] if x["partition"] == "calibration")
        locked = next(x for x in manifest["assignments"] if x["partition"] == "locked_test")
        invalid = copy.deepcopy(manifest)
        target = next(x for x in invalid["assignments"] if x["record_id"] == calibration["record_id"])
        target["canonical_group_id"] = locked["canonical_group_id"]
        invalid["split_content_sha256"] = data_pipeline.canonical_json_sha256(
            data_pipeline._manifest_identity_payload(invalid)
        )
        with self.assertRaises(data_pipeline.ManifestValidationError):
            data_pipeline.validate_split_manifest(invalid)

    def test_partition_loader_requests_only_calibration_rows(self) -> None:
        records = pd.DataFrame([{"record_id": "cal-1"}])
        calibration = pd.DataFrame([{"record_id": "cal-1"}])
        manifest = {"assignments": []}
        with tempfile.TemporaryDirectory() as temporary:
            canonical_path = Path(temporary) / "canonical.json"
            canonical_path.write_text(
                '{"canonical_data_manifest_sha256":"data-hash"}', encoding="utf-8"
            )
            args = types.SimpleNamespace(
                split_manifest=Path(temporary) / "split.json",
                canonical_data_manifest=canonical_path,
            )
            with mock.patch.object(
                calibrate_sbert_heads.training,
                "load_canonical_dataset",
                return_value=(
                    records,
                    {"canonical_data_manifest_sha256": "data-hash"},
                    {},
                    {"Non-biased": 0, "Biased": 1},
                    {"objective_style": 0, "opinionated_style": 1},
                ),
            ), mock.patch.object(
                calibrate_sbert_heads.data_pipeline, "load_split_manifest", return_value=manifest
            ), mock.patch.object(
                calibrate_sbert_heads, "validate_calibration_split_manifest"
            ), mock.patch.object(
                calibrate_sbert_heads.data_pipeline, "validate_split_manifest"
            ), mock.patch.object(
                calibrate_sbert_heads.data_pipeline, "validate_canonical_data_manifest"
            ), mock.patch.object(
                calibrate_sbert_heads.data_pipeline,
                "records_for_partition",
                return_value=calibration,
            ) as partition_rows, mock.patch.object(
                calibrate_sbert_heads.data_pipeline,
                "split_partition_assignment_sha256",
                return_value="assignment-hash",
            ):
                result = calibrate_sbert_heads._load_calibration_partition(args)
        partition_rows.assert_called_once_with(records, manifest, "calibration")
        self.assertIs(result["calibration_df"], calibration)


if __name__ == "__main__":
    unittest.main()
