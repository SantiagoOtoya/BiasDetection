from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_ROOT / "LLM-inference"
for import_path in (PROJECT_ROOT, LLM_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import artifact_manifest
import data_pipeline
import finetune_all_mpnet_babe as training


def load_infer_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "infer_bias_llm_binding", LLM_DIR / "infer_bias_llm.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["infer_bias_llm_binding"] = module
    spec.loader.exec_module(module)
    return module


infer_bias_llm = load_infer_module()


class InferenceCalibrationBindingTests(unittest.TestCase):
    bias_labels = {"Non-biased": 0, "Biased": 1}
    style_labels = {"objective_style": 0, "opinionated_style": 1}
    target_mapping = dict(data_pipeline.SOURCE_OPINION_TIER_TO_STYLE)

    def _region(self, threshold: float, direction: str, metric: str) -> dict:
        return {
            "enabled": True, "disabled_reason": None, "threshold": threshold,
            "direction": direction, "comparison": ">=" if direction == "upper" else "<=",
            "metric": metric, "target": 0.80, "confidence": 0.95,
            "minimum_predicted_support": 1, "supervised_support": 4,
            "reference_support": 2, "selected_support": 4, "correct_support": 4,
            "observed_metric": 1.0, "wilson_lower_bound": 0.80,
            "coverage": 1.0, "coverage_unit": "supervised_calibration_rows",
            "best_attempt": None,
        }

    def _calibrator(self) -> dict:
        return {
            "schema_version": "binary_calibrator/v1",
            "method": "temperature_scaling", "input_kind": "binary_logit",
            "output_kind": "positive_class_probability",
            "parameters": {"temperature": 1.0}, "fit_status": "optimized",
            "sample_support": {"supervised": 4, "positive": 2, "negative": 2},
            "diagnostics": {}, "nll_before": 0.5, "nll_after": 0.4,
        }

    def _create_fixture(self, root: Path) -> dict:
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"model")
        heads_path = root / "heads.pt"
        heads_path.write_bytes(b"heads")
        code_path = root / "training_code.py"
        code_path.write_text("print('bound')\n", encoding="utf-8")
        source = pd.DataFrame([
            {
                "uuid": str(i), "text": f"Sentence {i}",
                "news_link": f"https://example.org/{i}",
                "label_bias": "Biased" if i % 2 else "Non-biased",
                "label_opinion": "Somewhat factual but also opinionated" if i % 3 else "Entirely factual",
            }
            for i in range(20)
        ])
        records = data_pipeline.canonicalize_records(source)
        records, stats = data_pipeline.clean_and_dedupe_records(records)
        canonical = data_pipeline.build_canonical_data_manifest(records, stats)
        canonical_path = root / "canonical_data_manifest.json"
        data_pipeline.write_json(canonical_path, canonical)
        _assigned, split = data_pipeline.build_group_split_manifest(
            records, seed=19,
            canonical_data_manifest_sha256=canonical["canonical_data_manifest_sha256"],
            cleaning_stats=stats, created_at_utc="2026-07-10T00:00:00Z",
        )
        split_path = root / "split_manifest.json"
        data_pipeline.write_json(split_path, split)
        artifact = artifact_manifest.build_artifact_manifest(
            PROJECT_ROOT, model_path=model_dir, heads_path=heads_path,
            split_manifest_path=split_path,
            canonical_data_manifest_path=canonical_path,
            evaluation_scorecard_path=None, code_paths=[code_path],
            label_mappings={
                "bias_label2id": self.bias_labels,
                "opinion_style_label2id": self.style_labels,
            },
            head_type=training.INDEPENDENT_BINARY_HEADS_TYPE,
            head_architecture={"bias_head": {}, "opinion_style_head": {}},
            training_configuration={"seed": 19}, source_summary={"fixture": 1},
            model_excluded_relative_paths=("calibration.json", "artifact_manifest.json"),
        )
        files = artifact["core"]["files"]
        assignment_sha = data_pipeline.split_partition_assignment_sha256(split, "calibration")
        calibration_data = {
            "schema_version": "calibration_partition/v1", "partition_id": "calibration",
            "record_count": split["partition_summaries"]["calibration"]["record_count"],
            "records_sha256": "fixture-calibration-records",
            "split_partition_assignment_sha256": assignment_sha,
        }
        binding = {
            "artifact_manifest_sha256": artifact["artifact_manifest_sha256"],
            "model_sha256": files["model"]["sha256"],
            "heads_sha256": files["classification_heads"]["sha256"],
            "model_artifact_path": files["model"]["path"],
            "classification_heads_path": files["classification_heads"]["path"],
            "split_manifest_sha256": split["split_content_sha256"],
            "split_manifest_file_sha256": files["split_manifest"]["sha256"],
            "canonical_data_manifest_sha256": canonical["canonical_data_manifest_sha256"],
            "canonical_data_manifest_file_sha256": files["canonical_data_manifest"]["sha256"],
            "calibration_partition_id": "calibration",
            "calibration_records_sha256": calibration_data["records_sha256"],
            "calibration_partition_assignment_sha256": assignment_sha,
            "artifact_code_sha256": artifact_manifest.canonical_json_sha256(files["code"]),
            "calibration_code_sha256": artifact_manifest.sha256_path(PROJECT_ROOT / "calibrate_sbert_heads.py"),
            "checkpoint_schema_version": training.CLASSIFICATION_HEADS_SCHEMA_VERSION,
            "calibration_schema_version": "calibration/v3",
            "decision_policy_version": "classifier_decision/v1",
            "classifier_input_contract": data_pipeline.CLASSIFIER_INPUT_CONTRACT,
        }
        calibration = {
            "schema_version": "calibration/v3", "decision_policy_version": "classifier_decision/v1",
            "calibration_status": "valid", "artifact_binding": binding,
            "classifier_input_contract": data_pipeline.CLASSIFIER_INPUT_CONTRACT,
            "calibration_data": calibration_data,
            "head_types": {
                "classifier": training.INDEPENDENT_BINARY_HEADS_TYPE,
                "bias": training.BINARY_BIAS_HEAD_TYPE,
                "opinion_style": training.BINARY_OPINION_STYLE_HEAD_TYPE,
            },
            "label_mappings": {
                "bias_label2id": self.bias_labels,
                "opinion_style_label2id": self.style_labels,
                "opinion_style_target_mapping": self.target_mapping,
            },
            "calibrators": {"bias": self._calibrator(), "opinion_style": self._calibrator()},
            "thresholds": {
                "bias_no_clear_max": self._region(0.20, "lower", "negative_predictive_value"),
                "bias_clear_min": self._region(0.80, "upper", "precision"),
                "opinion_objective_max": self._region(0.25, "lower", "negative_predictive_value"),
                "opinion_opinionated_min": self._region(0.75, "upper", "precision"),
            },
        }
        return {
            "artifact": artifact, "artifact_path": model_dir / "artifact_manifest.json",
            "calibration": calibration, "calibration_path": model_dir / "calibration.json",
            "heads_path": heads_path, "model_dir": model_dir,
        }

    def _write_bound(self, fixture: dict, calibration: dict) -> None:
        fixture["calibration_path"].write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        bound = artifact_manifest.bind_calibration(
            fixture["artifact"], artifact_root=PROJECT_ROOT,
            calibration_path=fixture["calibration_path"],
            release_gate_results={"acceptance_gate": {"promotion_state": "eligible"}},
        )
        artifact_manifest.write_artifact_manifest(fixture["artifact_path"], bound)

    def _args(self, fixture: dict, mode: str = "calibrated") -> types.SimpleNamespace:
        return types.SimpleNamespace(
            selection_mode=mode, calibration_file=fixture["calibration_path"],
            artifact_manifest=fixture["artifact_path"], sbert_model_dir=fixture["model_dir"],
            classification_heads=fixture["heads_path"], bias_threshold=None,
            opinion_threshold=None, include_possible_bias=False,
        )

    def _resolve(self, fixture: dict) -> object:
        return infer_bias_llm.resolve_selection_config(
            self._args(fixture), self.bias_labels, self.style_labels,
            training.BINARY_OPINION_STYLE_HEAD_TYPE, heads_path=fixture["heads_path"],
            bias_head_type=training.BINARY_BIAS_HEAD_TYPE,
            classifier_head_type=training.INDEPENDENT_BINARY_HEADS_TYPE,
            checkpoint_schema_version=training.CLASSIFICATION_HEADS_SCHEMA_VERSION,
            opinion_style_target_mapping=self.target_mapping,
            classifier_input_contract=data_pipeline.CLASSIFIER_INPUT_CONTRACT,
        )

    def test_strict_binding_accepts_exact_v3_artifact(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            fixture = self._create_fixture(Path(temporary))
            self._write_bound(fixture, fixture["calibration"])
            config = self._resolve(fixture)
        self.assertEqual(config.effective_mode, "calibrated")
        self.assertIsNotNone(config.bias_decision)

    def test_strict_binding_rejects_model_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            fixture = self._create_fixture(Path(temporary))
            invalid = copy.deepcopy(fixture["calibration"])
            invalid["artifact_binding"]["model_sha256"] = "wrong"
            self._write_bound(fixture, invalid)
            with self.assertRaises(SystemExit):
                self._resolve(fixture)

    def test_strict_binding_rejects_label_and_head_mismatches(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            fixture = self._create_fixture(Path(temporary))
            invalid = copy.deepcopy(fixture["calibration"])
            invalid["label_mappings"]["bias_label2id"] = {"Non-biased": 1, "Biased": 0}
            self._write_bound(fixture, invalid)
            with self.assertRaises(SystemExit):
                self._resolve(fixture)

    def test_auto_missing_artifact_does_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            fixture = self._create_fixture(Path(temporary))
            args = self._args(fixture, "auto")
            args.calibration_file = fixture["model_dir"] / "missing.json"
            with self.assertRaises(SystemExit):
                infer_bias_llm.resolve_selection_config(
                    args, self.bias_labels, self.style_labels,
                    training.BINARY_OPINION_STYLE_HEAD_TYPE,
                )

    def test_argmax_requires_explicit_legacy_mode(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            fixture = self._create_fixture(Path(temporary))
            config = infer_bias_llm.resolve_selection_config(
                self._args(fixture, "argmax"), self.bias_labels, self.style_labels,
                training.LEGACY_FLAT_HEAD_TYPE,
            )
        self.assertEqual(config.effective_mode, "argmax")
        self.assertTrue(config.legacy_compatibility)


if __name__ == "__main__":
    unittest.main()
