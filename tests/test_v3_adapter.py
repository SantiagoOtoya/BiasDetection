from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
for import_path in (PROJECT_ROOT, BACKEND_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

# stack_loader is dependency-free; scoring and the schema mapping need pydantic.
import stack_loader

try:
    import fact_checker
    import scoring
    import v3_mapping
    from schemas import FactStatus

    HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover - environment without pydantic
    HAVE_PYDANTIC = False


@unittest.skipUnless(HAVE_PYDANTIC, "pydantic not installed in this interpreter")
class SentenceSignalV3Tests(unittest.TestCase):
    def test_blends_bias_and_style(self) -> None:
        self.assertAlmostEqual(scoring.sentence_signal_v3(1.0, 1.0), 1.0)
        self.assertAlmostEqual(scoring.sentence_signal_v3(0.0, 0.0), 0.0)
        self.assertAlmostEqual(scoring.sentence_signal_v3(1.0, 0.0), scoring.BIAS_WEIGHT)
        self.assertAlmostEqual(scoring.sentence_signal_v3(0.0, 1.0), scoring.OPINION_WEIGHT)

    def test_none_treated_as_zero(self) -> None:
        self.assertEqual(scoring.sentence_signal_v3(None, None), 0.0)

    def test_clamped_to_unit_interval(self) -> None:
        self.assertEqual(scoring.sentence_signal_v3(5.0, 5.0), 1.0)
        self.assertEqual(scoring.sentence_signal_v3(-1.0, -1.0), 0.0)


class StackLoaderTests(unittest.TestCase):
    def test_unknown_stack_rejected(self) -> None:
        with self.assertRaises(stack_loader.StackIsolationError):
            stack_loader._stack_dirs("v9")

    def test_v3_dirs_point_into_handoff(self) -> None:
        dirs = stack_loader._stack_dirs("v3")
        self.assertTrue(str(dirs[0]).endswith("Bias Encoder"))
        self.assertTrue((dirs[1] / "infer_bias_llm.py").exists())

    def test_v2_dirs_point_at_repo_root(self) -> None:
        (root,) = stack_loader._stack_dirs("v2")
        self.assertTrue((root / "infer_bias_llm.py").exists())


@unittest.skipUnless(HAVE_PYDANTIC, "pydantic not installed in this interpreter")
class StatusMappingTests(unittest.TestCase):
    def test_approved_mapping(self) -> None:
        self.assertEqual(fact_checker.map_assessor_status("supported"), FactStatus.verified)
        self.assertEqual(fact_checker.map_assessor_status("contradicted"), FactStatus.disputed)
        self.assertEqual(
            fact_checker.map_assessor_status("insufficient_evidence"), FactStatus.unverified
        )
        self.assertEqual(
            fact_checker.map_assessor_status("not_verifiable"), FactStatus.unverified
        )

    def test_never_emits_false_and_unknown_is_unverified(self) -> None:
        for status in ("false", "made_up", "", None):
            self.assertIn(
                fact_checker.map_assessor_status(status),
                {FactStatus.unverified, FactStatus.disputed, FactStatus.verified},
            )
        self.assertEqual(fact_checker.map_assessor_status("weird"), FactStatus.unverified)

    def test_fact_checks_from_v3_claims(self) -> None:
        claims = [
            {
                "claim": "Enrollment rose 3 percent last year.",
                "evidence_status": "supported",
                "rationale": "Matches the cited agency figure.",
                "citation_ids": ["E1", "E2"],
            },
            {
                "claim": "The program doubled costs.",
                "evidence_status": "contradicted",
                "rationale": "The cited budget shows a 4 percent increase.",
                "citation_ids": ["E3"],
            },
            {"claim": "", "evidence_status": "supported"},  # dropped: empty claim
        ]
        checks = fact_checker.fact_checks_from_v3_claims(claims)
        self.assertEqual(len(checks), 2)
        self.assertEqual(checks[0].status, FactStatus.verified)
        self.assertEqual(checks[0].evidence_ids, ["E1", "E2"])
        self.assertEqual(checks[1].status, FactStatus.disputed)


def make_record(**overrides):
    record = {
        "sentence_count": 3,
        "llm_revision": "abc123",
        "sentence_predictions": [
            {"index": 0, "text": "Neutral fact.", "p_bias": 0.1, "p_opinionated_style": 0.2},
            {"index": 1, "text": "Loaded claim.", "p_bias": 0.9, "p_opinionated_style": 0.8},
            {"index": 2, "text": "Another fact.", "p_bias": 0.2, "p_opinionated_style": 0.1},
        ],
        "selected_sentences": [
            {
                "index": 1,
                "text": "Loaded claim.",
                "bias_assessment": "clear_bias",
                "opinion_style": "opinionated_style",
                "p_bias": 0.9,
                "p_opinionated_style": 0.8,
                "selection_reasons": ["clear_bias"],
            }
        ],
        "selection_policy": {"effective_mode": "calibrated"},
        "abstention_summary": {"selected_sentence_count": 1},
        "evidence": [],
        "context_windows": [
            {
                "llm_report": "The wording 'loaded claim' frames the subject negatively.",
                "claims": [
                    {
                        "claim": "The program was cancelled.",
                        "evidence_status": "insufficient_evidence",
                        "rationale": "No cited source addresses this.",
                        "citation_ids": [],
                    }
                ],
            }
        ],
    }
    record.update(overrides)
    return record


@unittest.skipUnless(HAVE_PYDANTIC, "pydantic not installed in this interpreter")
class RecordToResponseTests(unittest.TestCase):
    RELEVANCE = {
        "total_sentences": 4,
        "sentences_after_relevance_filter": 3,
        "sentences_removed_as_irrelevant": 1,
        "removed_reason_counts": {"too_short": 1},
        "semantic_ran": False,
    }

    def response(self, record):
        return v3_mapping.record_to_response(
            record,
            self.RELEVANCE,
            mode="prompt-only",
            llm_used=False,
            evidence_mode="off",
        )

    def test_selected_sentence_mapping(self) -> None:
        response = self.response(make_record())
        self.assertEqual(len(response.selected_sentences), 1)
        s = response.selected_sentences[0]
        self.assertEqual(s.bias_assessment, "clear_bias")
        self.assertEqual(s.opinion_style, "opinionated_style")
        self.assertEqual(s.bias_label, "clear_bias")
        self.assertAlmostEqual(s.bias_probability, 0.9)

    def test_score_uses_all_predictions(self) -> None:
        response = self.response(make_record())
        expected_signals = [
            scoring.sentence_signal_v3(0.1, 0.2),
            scoring.sentence_signal_v3(0.9, 0.8),
            scoring.sentence_signal_v3(0.2, 0.1),
        ]
        self.assertAlmostEqual(
            response.overall_score, scoring.compute_score(expected_signals)
        )
        self.assertIn("relevant sentence", response.score_caption)

    def test_report_combines_window_reports(self) -> None:
        response = self.response(make_record())
        self.assertIn("frames the subject negatively", response.report)

    def test_fact_checks_use_claim_assessments(self) -> None:
        response = self.response(make_record())
        self.assertEqual(len(response.fact_checks), 1)
        self.assertEqual(response.fact_checks[0].status, FactStatus.unverified)

    def test_no_claims_falls_back_to_conservative(self) -> None:
        record = make_record(context_windows=[{"llm_report": None, "claims": []}])
        response = self.response(record)
        self.assertTrue(
            all(fc.status == FactStatus.unverified for fc in response.fact_checks)
        )

    def test_meta_carries_stack_and_relevance(self) -> None:
        response = self.response(make_record())
        self.assertEqual(response.meta["stack"], "v3")
        self.assertEqual(response.meta["relevance"], self.RELEVANCE)
        self.assertEqual(response.meta["evidence_mode"], "off")
        self.assertFalse(response.meta["evidence_enabled"])

    def test_no_selected_sentences_reports_clean_page(self) -> None:
        record = make_record(selected_sentences=[], context_windows=[])
        response = self.response(record)
        self.assertIn("No clearly biased wording", response.report)


if __name__ == "__main__":
    unittest.main()
