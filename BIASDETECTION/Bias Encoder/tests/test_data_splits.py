from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_pipeline


class DataSplitTests(unittest.TestCase):
    def canonical_records(self) -> pd.DataFrame:
        rows = []
        opinions = [
            "Entirely factual",
            "Somewhat factual but also opinionated",
            "Expresses writer's opinion",
        ]
        for index in range(15):
            rows.append(
                {
                    "uuid": f"babe-{index}",
                    "text": f"Sentence for article {index}.",
                    "news_link": f"https://Example.com/article/{index}?utm_source=test",
                    "label_bias": "Biased" if index % 2 else "Non-biased",
                    "label_opinion": opinions[index % len(opinions)],
                    "outlet": "example",
                    "topic": "policy",
                }
            )
        rows.extend(
            [
                {
                    "uuid": "fallback-a",
                    "text": "First sentence in an URL-less article.",
                    "article": "URL-less article body with two sentences.",
                    "label_bias": "Biased",
                    "label_opinion": "Somewhat factual but also opinionated",
                },
                {
                    "uuid": "fallback-b",
                    "text": "Second sentence in an URL-less article.",
                    "article": "URL-less article body with two sentences.",
                    "label_bias": "Non-biased",
                    "label_opinion": "Entirely factual",
                },
                {
                    "uuid": "duplicate",
                    "text": "Sentence for article 0.",
                    "news_link": "https://example.com/reprint/0",
                    "label_bias": "Non-biased",
                    "label_opinion": "Entirely factual",
                },
            ]
        )
        babe = data_pipeline.canonicalize_records(pd.DataFrame(rows), source_dataset="BABE")
        basil = data_pipeline.canonicalize_records(
            pd.DataFrame(
                [
                    {
                        "uuid": "basil-a",
                        "text": "BASIL story sentence one.",
                        "event_id": "Story-9",
                        "label_bias": "Biased",
                        "label_opinion": "Entirely factual",
                    },
                    {
                        "uuid": "basil-b",
                        "text": "BASIL story sentence two.",
                        "event_id": "Story-9",
                        "label_bias": "Non-biased",
                        "label_opinion": "Somewhat factual but also opinionated",
                    },
                ]
            ),
            source_dataset="BASIL",
        )
        cleaned, _stats = data_pipeline.clean_and_dedupe_records(
            pd.concat([babe, basil], ignore_index=True)
        )
        return cleaned

    def build_manifest(self):
        records = self.canonical_records()
        data_manifest = data_pipeline.build_canonical_data_manifest(records, {})
        return data_pipeline.build_group_split_manifest(
            records,
            seed=23,
            canonical_data_manifest_sha256=data_manifest["canonical_data_manifest_sha256"],
            cleaning_stats={},
            created_at_utc="2026-07-10T00:00:00Z",
        )

    def test_canonical_url_article_fallback_and_basil_event_grouping(self) -> None:
        records = self.canonical_records()
        url_record = records[records["source_record_id"] == "babe-1"].iloc[0]
        self.assertEqual(
            url_record["canonical_url"], "https://example.com/article/1"
        )
        fallback = records[records["source_record_id"] == "fallback-a"].iloc[0]
        self.assertEqual(fallback["grouping_method"], "article_hash")
        second_fallback = records[records["source_record_id"] == "fallback-b"].iloc[0]
        self.assertEqual(fallback["canonical_group_id"], second_fallback["canonical_group_id"])
        basil_groups = set(
            records.loc[records["source_dataset"] == "BASIL", "canonical_group_id"]
        )
        self.assertEqual(basil_groups, {"basil:event:story-9"})

    def test_global_deduplication_happens_before_split(self) -> None:
        records = self.canonical_records()
        duplicated_text = records[records["text"] == "Sentence for article 0."]
        self.assertEqual(len(duplicated_text), 1)

    def test_group_event_and_text_hashes_do_not_overlap(self) -> None:
        assigned, manifest = self.build_manifest()
        self.assertFalse(manifest["leakage_report"]["has_overlap"])
        for left_index, left in enumerate(data_pipeline.PARTITIONS):
            for right in data_pipeline.PARTITIONS[left_index + 1 :]:
                left_rows = assigned[assigned["partition"] == left]
                right_rows = assigned[assigned["partition"] == right]
                for column in ("canonical_group_id", "event_id", "text_hash"):
                    left_values = set(left_rows[column].dropna())
                    right_values = set(right_rows[column].dropna())
                    self.assertFalse(left_values.intersection(right_values), column)
        basil_partitions = set(
            assigned.loc[assigned["source_dataset"] == "BASIL", "partition"]
        )
        self.assertEqual(len(basil_partitions), 1)

    def test_split_reproduction_uses_stable_content_hash(self) -> None:
        first_assigned, first_manifest = self.build_manifest()
        second_assigned, second_manifest = self.build_manifest()
        self.assertEqual(
            first_manifest["split_content_sha256"], second_manifest["split_content_sha256"]
        )
        self.assertEqual(
            first_assigned[["record_id", "partition"]].sort_values("record_id").to_dict("records"),
            second_assigned[["record_id", "partition"]].sort_values("record_id").to_dict("records"),
        )

    def test_canonical_data_manifest_hash_and_partition_fingerprint_validate(self) -> None:
        records = self.canonical_records()
        data_manifest = data_pipeline.build_canonical_data_manifest(records, {})
        data_pipeline.validate_canonical_data_manifest(data_manifest)
        _assigned, split_manifest = data_pipeline.build_group_split_manifest(
            records,
            seed=23,
            canonical_data_manifest_sha256=data_manifest[
                "canonical_data_manifest_sha256"
            ],
            cleaning_stats={},
            created_at_utc="2026-07-10T00:00:00Z",
        )
        fingerprint = data_pipeline.split_partition_assignment_sha256(
            split_manifest, "calibration"
        )
        self.assertEqual(
            fingerprint,
            data_pipeline.split_partition_assignment_sha256(
                split_manifest, "calibration"
            ),
        )
        invalid = copy.deepcopy(data_manifest)
        invalid["records"][0]["text_hash"] = "tampered"
        with self.assertRaises(data_pipeline.ManifestValidationError):
            data_pipeline.validate_canonical_data_manifest(invalid)

    def test_manifest_validation_rejects_unknown_partition(self) -> None:
        _assigned, manifest = self.build_manifest()
        invalid = copy.deepcopy(manifest)
        invalid["assignments"][0]["partition"] = "validation"
        with self.assertRaises(data_pipeline.ManifestValidationError):
            data_pipeline.validate_split_manifest(invalid)

    def test_legacy_manifest_reports_but_allows_explicit_leakage(self) -> None:
        all_records = self.canonical_records()
        records = pd.concat(
            [
                all_records[all_records["source_dataset"] == "BASIL"],
                all_records[all_records["source_dataset"] == "BABE"].iloc[:4],
            ],
            ignore_index=True,
        )
        data_manifest = data_pipeline.build_canonical_data_manifest(records, {})
        assignments = {
            str(row["record_id"]): ("train" if index % 2 else "locked_test")
            for index, (_, row) in enumerate(records.iterrows())
        }
        _assigned, manifest = data_pipeline.build_legacy_split_manifest(
            records,
            assignments,
            seed=1,
            canonical_data_manifest_sha256=data_manifest["canonical_data_manifest_sha256"],
            cleaning_stats={},
        )
        self.assertTrue(manifest["leakage_report"]["has_overlap"])
        data_pipeline.validate_split_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
