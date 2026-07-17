from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_pipeline
import finetune_all_mpnet_babe as training


class OpinionPreprocessingAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        training.pd = pd

    def parse_args(
        self,
        root: Path,
        *,
        split_strategy: str = "group",
        include_no_agreement: bool = False,
        classifier_mode: str = training.PRODUCTION_CLASSIFIER_MODE,
    ):
        arguments = [
            "finetune_all_mpnet_babe.py",
            "--data-dir",
            str(root),
            "--output-dir",
            str(root / "output"),
            "--split-manifest",
            str(root / "split.json"),
            "--canonical-data-manifest",
            str(root / "canonical.json"),
            "--split-strategy",
            split_strategy,
            "--missing-label-policy",
            "mask",
            "--classifier-mode",
            classifier_mode,
            "--validation-size",
            "0.15",
            "--calibration-size",
            "0.15",
            "--test-size",
            "0.20",
            "--seed",
            "23",
        ]
        if split_strategy == "legacy":
            arguments.extend(["--leakage-policy", "warn"])
        if include_no_agreement:
            arguments.append("--include-no-agreement")
        with mock.patch.object(sys, "argv", arguments):
            return training.parse_args()

    @staticmethod
    def write_train_csv(root: Path, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(root / "train.csv", index=False, sep=";")

    @staticmethod
    def opinion_rows(count: int = 36) -> list[dict[str, object]]:
        opinions = [
            "Entirely factual",
            "Somewhat factual but also opinionated",
            "Expresses writer's opinion",
        ]
        return [
            {
                "uuid": f"record-{index}",
                "text": f"Independent sentence {index}.",
                "news_link": f"https://example.test/article/{index}",
                "label_bias": "Biased" if index % 2 else "Non-biased",
                "label_opinion": opinions[index % len(opinions)],
            }
            for index in range(count)
        ]

    def test_opinion_normalization_handles_apostrophes_whitespace_and_case(self) -> None:
        cases = {
            " Entirely   factual ": ("Entirely factual", 0),
            " SOMEWHAT factual but also OPINIONATED ": (
                "Somewhat factual but also opinionated",
                1,
            ),
            "Expresses writer's opinion": ("Expresses writer's opinion", 2),
            "Expresses writer’s opinion": ("Expresses writer's opinion", 2),
            "Expresses writer‘S opinion": ("Expresses writer's opinion", 2),
        }
        for raw, (expected_label, expected_id) in cases.items():
            normalized = data_pipeline.normalize_opinion_label(raw)
            self.assertEqual(normalized, expected_label)
            self.assertEqual(data_pipeline.opinion_label_value(normalized), expected_id)
            self.assertEqual(
                data_pipeline.opinion_style_label_value(normalized),
                0 if expected_id == 0 else 1,
            )

    def test_mask_policy_keeps_valid_tiers_and_masks_invalid_opinion_values(self) -> None:
        rows = [
            {
                "uuid": "factual",
                "text": "Factual sentence.",
                "news_link": "https://example.test/factual",
                "label_bias": "Non-biased",
                "label_opinion": "Entirely factual",
            },
            {
                "uuid": "middle",
                "text": "Middle sentence.",
                "news_link": "https://example.test/middle",
                "label_bias": "Biased",
                "label_opinion": "  SOMEWHAT factual but also OPINIONATED  ",
            },
            {
                "uuid": "writer",
                "text": "Writer sentence.",
                "news_link": "https://example.test/writer",
                "label_bias": "Non-biased",
                "label_opinion": "Expresses writer’s opinion",
            },
            {
                "uuid": "missing",
                "text": "Missing sentence.",
                "news_link": "https://example.test/missing",
                "label_bias": "Biased",
                "label_opinion": None,
            },
            {
                "uuid": "unknown",
                "text": "Unknown sentence.",
                "news_link": "https://example.test/unknown",
                "label_bias": "Non-biased",
                "label_opinion": "Undocumented opinion label",
            },
            {
                "uuid": "no-agreement",
                "text": "No agreement sentence.",
                "news_link": "https://example.test/no-agreement",
                "label_bias": "Biased",
                "label_opinion": "No agreement",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_train_csv(root, rows)
            included_args = self.parse_args(root)
            records, _manifest, _stats, _bias_mapping, opinion_mapping = (
                training.load_canonical_dataset(included_args)
            )
            labels_by_record = records.set_index("source_record_id")[
                "opinion_style_label_id"
            ]
            self.assertEqual(int(labels_by_record["factual"]), 0)
            self.assertEqual(int(labels_by_record["middle"]), 1)
            self.assertEqual(int(labels_by_record["writer"]), 1)
            for record_id in ("missing", "unknown", "no-agreement"):
                self.assertTrue(pd.isna(labels_by_record[record_id]), record_id)
            self.assertEqual(opinion_mapping, training.OPINION_STYLE_LABEL2ID)
            self.assertEqual(
                int(records.set_index("source_record_id").at["no-agreement", "bias_label_id"]),
                1,
            )

            excluded_args = self.parse_args(
                root,
                include_no_agreement=False,
                classifier_mode=training.LEGACY_HIERARCHICAL_CLASSIFIER_MODE,
            )
            excluded_records, *_ = training.load_canonical_dataset(excluded_args)
            self.assertNotIn(
                "no-agreement", set(excluded_records["source_record_id"])
            )

    def test_group_split_balances_canonical_opinion_ids_and_is_group_disjoint(self) -> None:
        records = data_pipeline.canonicalize_records(
            pd.DataFrame(self.opinion_rows()), source_dataset="BABE"
        )
        records, _stats = data_pipeline.clean_and_dedupe_records(records)
        data_manifest = data_pipeline.build_canonical_data_manifest(records, {})
        first_assigned, first_manifest = data_pipeline.build_group_split_manifest(
            records,
            seed=23,
            canonical_data_manifest_sha256=data_manifest[
                "canonical_data_manifest_sha256"
            ],
            cleaning_stats={},
            created_at_utc="2026-07-11T00:00:00Z",
        )
        second_assigned, second_manifest = data_pipeline.build_group_split_manifest(
            records,
            seed=23,
            canonical_data_manifest_sha256=data_manifest[
                "canonical_data_manifest_sha256"
            ],
            cleaning_stats={},
            created_at_utc="2026-07-11T00:00:00Z",
        )

        train_ids = set(
            first_assigned.loc[
                first_assigned["partition"] == "train", "opinion_label"
            ].astype(int)
        )
        self.assertEqual(train_ids, {0, 1, 2})
        self.assertFalse(first_manifest["leakage_report"]["has_overlap"])
        for left_index, left in enumerate(data_pipeline.PARTITIONS):
            left_groups = set(
                first_assigned.loc[
                    first_assigned["partition"] == left, "canonical_group_id"
                ]
            )
            for right in data_pipeline.PARTITIONS[left_index + 1 :]:
                right_groups = set(
                    first_assigned.loc[
                        first_assigned["partition"] == right, "canonical_group_id"
                    ]
                )
                self.assertFalse(left_groups.intersection(right_groups))
        self.assertEqual(
            first_manifest["split_content_sha256"],
            second_manifest["split_content_sha256"],
        )
        self.assertEqual(
            first_assigned[["record_id", "partition"]]
            .sort_values("record_id")
            .to_dict("records"),
            second_assigned[["record_id", "partition"]]
            .sort_values("record_id")
            .to_dict("records"),
        )

    def test_preparation_logs_all_opinion_count_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_train_csv(root, self.opinion_rows())
            args = self.parse_args(root)
            with self.assertLogs("finetune_all_mpnet_babe", level="INFO") as logs:
                prepared = training.prepare_data_and_splits(args)

        self.assertEqual(
            set(
                prepared["partitions"]["train"]["opinion_style_label_id"]
                .dropna()
                .astype(int)
            ),
            {0, 1},
        )
        output = "\n".join(logs.output)
        for expected in (
            "Opinion labels raw input:",
            "Opinion labels after canonical normalization:",
            "Opinion-style targets after normalization/masking/dedupe:",
            "Opinion-style targets train partition:",
            "Opinion-style targets development partition:",
            "Opinion-style targets calibration partition:",
            "Opinion-style targets locked_test partition:",
        ):
            self.assertIn(expected, output)

    def test_hierarchical_train_tier_failure_reports_all_partition_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_train_csv(root, self.opinion_rows(18))
            args = self.parse_args(
                root,
                split_strategy="legacy",
                classifier_mode=training.LEGACY_HIERARCHICAL_CLASSIFIER_MODE,
            )
            records, data_manifest, cleaning_stats, _bias_mapping, _opinion_mapping = (
                training.load_canonical_dataset(args)
            )
            middle_seen = 0
            assignments: dict[str, str] = {}
            for _, row in records.iterrows():
                opinion_id = int(row["opinion_label"])
                if opinion_id == 0:
                    partition = "train"
                elif opinion_id == 1:
                    partition = "development" if middle_seen % 2 else "calibration"
                    middle_seen += 1
                else:
                    partition = "locked_test"
                assignments[str(row["record_id"])] = partition
            _assigned, manifest = data_pipeline.build_legacy_split_manifest(
                records,
                assignments,
                seed=args.seed,
                canonical_data_manifest_sha256=data_manifest[
                    "canonical_data_manifest_sha256"
                ],
                cleaning_stats=cleaning_stats,
            )
            data_pipeline.write_json(args.split_manifest, manifest)

            with self.assertRaises(ValueError) as raised:
                training.prepare_data_and_splits(args)

        message = str(raised.exception)
        self.assertIn("Hierarchical opinion training requires every tier", message)
        self.assertIn("Partition diagnostics:", message)
        self.assertIn("groups=", message)
        for partition in data_pipeline.PARTITIONS:
            self.assertIn(f"{partition}:", message)


if __name__ == "__main__":
    unittest.main()
