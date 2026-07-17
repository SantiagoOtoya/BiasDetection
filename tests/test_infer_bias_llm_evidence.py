from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evidence_retrieval


def load_infer_module() -> types.ModuleType:
    module_path = PROJECT_ROOT / "infer_bias_llm.py"
    spec = importlib.util.spec_from_file_location("infer_bias_llm", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load infer_bias_llm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["infer_bias_llm"] = module
    spec.loader.exec_module(module)
    return module


infer_bias_llm = load_infer_module()


def selected_prediction(index: int, text: str):
    return infer_bias_llm.SentencePrediction(
        index=index,
        text=text,
        bias_label="Biased",
        bias_probability=0.9,
        bias_probabilities={"Non-biased": 0.1, "Biased": 0.9},
        opinion_label="Expresses writer's opinion",
        opinion_probability=0.8,
        opinion_probabilities={"Entirely factual": 0.2, "Expresses writer's opinion": 0.8},
        selected=True,
    )


def evidence_item() -> evidence_retrieval.EvidenceItem:
    return evidence_retrieval.EvidenceItem(
        title="Federal Reserve data",
        url="https://www.federalreserve.gov/releases/z1/dataviz/dfa/",
        domain="federalreserve.gov",
        source_type="primary_official",
        published_date=None,
        snippet="Distributional Financial Accounts data.",
        query="Federal Reserve wealth data",
        retrieval_reason="selected_sentence",
    )


class InferBiasLlmEvidenceTests(unittest.TestCase):
    def test_build_prompt_omits_evidence_section_when_empty(self) -> None:
        prediction = selected_prediction(0, "The claim uses charged framing.")
        window = infer_bias_llm.ContextWindow(
            start_index=0,
            end_index=0,
            selected_sentence_indices=[0],
            text=prediction.text,
        )

        prompt = infer_bias_llm.build_prompt(window, {0: prediction})

        self.assertIn("sentences_for_review", prompt)
        self.assertNotIn("trusted_external_evidence", prompt)

    def test_build_prompt_includes_trusted_evidence_section(self) -> None:
        prediction = selected_prediction(
            0,
            "Federal Reserve data show the top 1 percent holds a large share of wealth.",
        )
        window = infer_bias_llm.ContextWindow(
            start_index=0,
            end_index=0,
            selected_sentence_indices=[0],
            text=prediction.text,
        )

        prompt = infer_bias_llm.build_prompt(window, {0: prediction}, [evidence_item()])

        self.assertIn("trusted_external_evidence", prompt)
        self.assertIn('"citation_id": "E1"', prompt)
        self.assertIn("federalreserve.gov", prompt)

    def test_analyze_article_records_not_requested_when_evidence_disabled(self) -> None:
        prediction = selected_prediction(0, "The proposal is a reckless giveaway.")
        args = types.SimpleNamespace(
            batch_size=16,
            context_sentences=0,
            max_windows_per_article=0,
            sbert_model_dir="model",
            llm_model="llm",
            enable_evidence=False,
        )

        with mock.patch.object(
            infer_bias_llm,
            "classify_sentences",
            return_value=[prediction],
        ):
            record = infer_bias_llm.analyze_article(
                article=infer_bias_llm.ArticleRecord(
                    article_id="a1",
                    text=prediction.text,
                    metadata={},
                ),
                model=None,
                bias_head=None,
                opinion_head=None,
                bias_label2id={},
                opinion_label2id={},
                device=None,
                args=args,
                llm=None,
            )

        window = record["context_windows"][0]
        self.assertEqual(window["evidence_status"], "not_requested")
        self.assertEqual(window["evidence_items"], [])

    def test_analyze_article_records_found_evidence(self) -> None:
        prediction = selected_prediction(
            0,
            "Federal Reserve data show the top 1 percent holds a large share of wealth.",
        )
        args = types.SimpleNamespace(
            batch_size=16,
            context_sentences=0,
            max_windows_per_article=0,
            sbert_model_dir="model",
            llm_model="llm",
            enable_evidence=True,
            evidence_provider="brave",
            max_evidence_items=5,
            evidence_timeout_seconds=1,
        )
        result = evidence_retrieval.EvidenceResult(status="found", items=[evidence_item()])

        with mock.patch.object(
            infer_bias_llm,
            "classify_sentences",
            return_value=[prediction],
        ):
            with mock.patch.object(
                infer_bias_llm.evidence_retrieval,
                "retrieve_evidence",
                return_value=result,
            ):
                record = infer_bias_llm.analyze_article(
                    article=infer_bias_llm.ArticleRecord(
                        article_id="a1",
                        text=prediction.text,
                        metadata={},
                    ),
                    model=None,
                    bias_head=None,
                    opinion_head=None,
                    bias_label2id={},
                    opinion_label2id={},
                    device=None,
                    args=args,
                    llm=None,
                )

        window = record["context_windows"][0]
        self.assertEqual(window["evidence_status"], "found")
        self.assertEqual(window["evidence_items"][0]["citation_id"], "E1")
        self.assertEqual(window["evidence_items"][0]["domain"], "federalreserve.gov")
        self.assertIn("trusted_external_evidence", window["prompt"])


if __name__ == "__main__":
    unittest.main()
