from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_ROOT / "LLM-inference"
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))

import evidence_retrieval


class EvidenceRetrievalTests(unittest.TestCase):
    def test_classifies_only_trusted_sources(self) -> None:
        self.assertEqual(
            evidence_retrieval.classify_trusted_source("https://www.cdc.gov/data"),
            "primary_official",
        )
        self.assertEqual(
            evidence_retrieval.classify_trusted_source(
                "https://pubmed.ncbi.nlm.nih.gov/123456/"
            ),
            "empirical_research",
        )
        self.assertEqual(
            evidence_retrieval.classify_trusted_source(
                "https://www.reuters.com/world/example"
            ),
            "wire_factual",
        )
        self.assertIsNone(
            evidence_retrieval.classify_trusted_source("https://example.com/story")
        )
        self.assertIsNone(
            evidence_retrieval.classify_trusted_source("https://twitter.com/post")
        )

    def test_build_evidence_queries_dedupes_and_compacts(self) -> None:
        queries = evidence_retrieval.build_evidence_queries(
            selected_sentence_texts=[
                "Federal Reserve data show the top 1 percent holds a large share of household wealth.",
                "Federal Reserve data show the top 1 percent holds a large share of household wealth.",
            ],
            context_text="The article discusses household wealth and federal tax policy.",
        )

        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[0].source, "selected_sentence")
        self.assertIn("Federal", queries[0].text)
        self.assertLessEqual(len(queries[0].text.split()), 14)

    def test_retrieve_evidence_requires_api_key(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            result = evidence_retrieval.retrieve_evidence(
                selected_sentence_texts=["A factual claim that needs evidence."],
                context_text="A factual claim that needs evidence.",
                provider="brave",
                max_items=5,
                timeout_seconds=1,
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.items, [])
        self.assertIn("BRAVE_SEARCH_API_KEY", result.error or "")

    def test_retrieve_evidence_filters_and_caps_mocked_results(self) -> None:
        item = evidence_retrieval.EvidenceItem(
            title="Federal Reserve data",
            url="https://www.federalreserve.gov/releases/z1/dataviz/dfa/",
            domain="federalreserve.gov",
            source_type="primary_official",
            published_date=None,
            snippet="Distributional Financial Accounts data.",
            query="Federal Reserve wealth data",
            retrieval_reason="selected_sentence",
        )

        with mock.patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "token"}):
            with mock.patch.object(
                evidence_retrieval,
                "search_brave",
                return_value=[item, item],
            ):
                result = evidence_retrieval.retrieve_evidence(
                    selected_sentence_texts=["Federal Reserve wealth data"],
                    context_text="Federal Reserve wealth data",
                    provider="brave",
                    max_items=1,
                    timeout_seconds=1,
                )

        self.assertEqual(result.status, "found")
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].domain, "federalreserve.gov")


if __name__ == "__main__":
    unittest.main()
