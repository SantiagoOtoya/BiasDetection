from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import artifact_manifest


class ArtifactManifestTests(unittest.TestCase):
    def create_manifest(self, root: Path) -> tuple[dict, Path]:
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"model")
        (root / "heads.pt").write_bytes(b"heads")
        (root / "split_manifest.json").write_text("{}", encoding="utf-8")
        (root / "canonical_data_manifest.json").write_text("{}", encoding="utf-8")
        (root / "scorecard.json").write_text("{}", encoding="utf-8")
        (root / "training.py").write_text("print('ok')\n", encoding="utf-8")
        manifest = artifact_manifest.build_artifact_manifest(
            root,
            model_path=model_dir,
            heads_path=root / "heads.pt",
            split_manifest_path=root / "split_manifest.json",
            canonical_data_manifest_path=root / "canonical_data_manifest.json",
            evaluation_scorecard_path=root / "scorecard.json",
            code_paths=[root / "training.py"],
            label_mappings={"bias_label2id": {"Non-biased": 0, "Biased": 1}},
            head_type="legacy_flat_v1",
            head_architecture={"opinion_head": "one_hidden_layer_mlp"},
            training_configuration={"seed": 42},
            source_summary={"BABE": 1},
        )
        return manifest, root

    def test_manifest_hashes_validate_and_detect_file_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, root = self.create_manifest(Path(temporary))
            artifact_manifest.validate_artifact_manifest(
                manifest, artifact_root=root, verify_files=True
            )
            (root / "heads.pt").write_bytes(b"tampered")
            with self.assertRaises(artifact_manifest.ArtifactManifestValidationError):
                artifact_manifest.validate_artifact_manifest(
                    manifest, artifact_root=root, verify_files=True
                )

    def test_calibration_binding_preserves_core_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, root = self.create_manifest(Path(temporary))
            calibration = root / "calibration.json"
            calibration.write_text("{}", encoding="utf-8")
            bound = artifact_manifest.bind_calibration(
                manifest,
                artifact_root=root,
                calibration_path=calibration,
            )
            self.assertEqual(
                bound["artifact_manifest_sha256"], manifest["artifact_manifest_sha256"]
            )
            self.assertEqual(bound["promotion_state"], "pending_locked_test")
            self.assertEqual(bound["frozen_artifact"]["schema_version"], "frozen_artifact/v1")
            self.assertIsNone(bound["locked_test_evaluation"])
            artifact_manifest.validate_artifact_manifest(
                bound, artifact_root=root, verify_files=True
            )

            scorecard = root / "locked_test_scorecard.json"
            scorecard.write_text("{}", encoding="utf-8")
            evaluated = artifact_manifest.bind_locked_test_evaluation(
                bound,
                artifact_root=root,
                scorecard_path=scorecard,
                release_gate_results={
                    "acceptance_gate": {
                        "passed": True,
                        "failures": [],
                        "promotion_state": "eligible",
                    }
                },
            )
            self.assertEqual(evaluated["promotion_state"], "eligible")
            self.assertEqual(
                evaluated["frozen_artifact"]["frozen_artifact_sha256"],
                bound["frozen_artifact"]["frozen_artifact_sha256"],
            )
            with self.assertRaises(artifact_manifest.ArtifactManifestValidationError):
                artifact_manifest.bind_locked_test_evaluation(
                    evaluated,
                    artifact_root=root,
                    scorecard_path=scorecard,
                    release_gate_results=evaluated["release_gate_results"],
                )


if __name__ == "__main__":
    unittest.main()
