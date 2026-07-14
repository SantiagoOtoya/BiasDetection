from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_ROOT / "LLM-inference"
for path in (PROJECT_ROOT, LLM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evidence_assessment
import evidence_retrieval


def claim() -> evidence_assessment.ExtractedClaim:
    return evidence_assessment.ExtractedClaim(
        claim_id="a:sentence:0:claim:1",
        sentence_index=0,
        sentence_text="The agency reported 10 cases.",
        context_text="The agency reported 10 cases yesterday.",
        text="The agency reported 10 cases.",
    )


def item(*, provider: str, claim_ids: tuple[str, ...] = ()) -> evidence_retrieval.EvidenceItem:
    return evidence_retrieval.EvidenceItem(
        title="Agency report",
        url="https://agency.gov/report",
        domain="agency.gov",
        source_type="primary_official",
        published_date="2026-01-01",
        snippet="The agency recorded 10 cases.",
        query="agency 10 cases",
        retrieval_reason="claim",
        claim_ids=claim_ids,
        canonical_url="https://agency.gov/report",
        provenance=evidence_retrieval.EvidenceProvenance(
            provider=provider,
            query="agency 10 cases",
            query_source="claim",
            collection="trusted" if provider == "qdrant" else None,
            document_id="d1" if provider == "qdrant" else None,
            chunk_id="c1" if provider == "qdrant" else None,
            point_id="p1" if provider == "qdrant" else None,
            score=0.999,
        ),
    )


class EvidenceAssessmentTests(unittest.TestCase):
    def test_claim_extraction_distinguishes_verifiable_and_opinion(self) -> None:
        def generate(_system: str, _prompt: str) -> str:
            return (
                '[{"sentence_index":0,"verifiability":"verifiable",'
                '"claims":["The agency reported 10 cases."]},'
                '{"sentence_index":1,"verifiability":"not_verifiable","claims":[]}]'
            )

        result = evidence_assessment.extract_verifiable_claims(
            article_id="a",
            candidates=[
                {"sentence_index": 0, "sentence": "The agency reported 10 cases.", "context": ""},
                {"sentence_index": 1, "sentence": "This plan is outrageous.", "context": ""},
            ],
            generate=generate,
        )
        self.assertEqual(result[0].verifiability, "verifiable")
        self.assertEqual(result[0].claims[0].claim_id, "a:sentence:0:claim:1")
        self.assertEqual(result[1].verifiability, "not_verifiable")
        self.assertEqual(result[1].claims, ())

    def test_search_snippet_cannot_produce_strong_status(self) -> None:
        registry = evidence_assessment.CitationRegistry()
        citations = registry.register([item(provider="brave", claim_ids=(claim().claim_id,))])
        called = False

        def generate(_system: str, _prompt: str) -> str:
            nonlocal called
            called = True
            return "[]"

        assessments = evidence_assessment.assess_claims(
            claims=[claim()], retrieval_status="found", citations=citations, generate=generate
        )
        self.assertFalse(called)
        self.assertEqual(assessments[0].evidence_status, "insufficient_evidence")
        self.assertEqual(assessments[0].citation_ids, ())

    def test_direct_corpus_text_and_valid_citation_can_support(self) -> None:
        citations = evidence_assessment.CitationRegistry().register(
            [item(provider="qdrant", claim_ids=(claim().claim_id,))]
        )
        seen_prompt = ""

        def generate(_system: str, prompt: str) -> str:
            nonlocal seen_prompt
            seen_prompt = prompt
            return (
                '[{"claim_id":"a:sentence:0:claim:1","evidence_status":"supported",'
                '"rationale":"The excerpt directly reports the count.","citation_ids":["E1"]}]'
            )

        assessments = evidence_assessment.assess_claims(
            claims=[claim()], retrieval_status="found", citations=citations, generate=generate
        )
        self.assertEqual(assessments[0].evidence_status, "supported")
        self.assertEqual(assessments[0].citation_ids, ("E1",))
        self.assertNotIn("0.999", seen_prompt)

    def test_unknown_citation_downgrades_strong_status(self) -> None:
        citations = evidence_assessment.CitationRegistry().register(
            [item(provider="qdrant", claim_ids=(claim().claim_id,))]
        )

        def generate(_system: str, _prompt: str) -> str:
            return (
                '[{"claim_id":"a:sentence:0:claim:1","evidence_status":"contradicted",'
                '"rationale":"Contradicted.","citation_ids":["E999"]}]'
            )

        assessment = evidence_assessment.assess_claims(
            claims=[claim()], retrieval_status="found", citations=citations, generate=generate
        )[0]
        self.assertEqual(assessment.evidence_status, "insufficient_evidence")
        self.assertEqual(assessment.citation_ids, ())

    def test_retrieval_failure_has_no_assessment_or_citations(self) -> None:
        assessment = evidence_assessment.assess_claims(
            claims=[claim()], retrieval_status="error", citations=[], generate=lambda *_: "[]"
        )[0]
        self.assertEqual(assessment.evidence_status, "not_assessed")
        self.assertEqual(assessment.citation_ids, ())

    def test_sentence_status_is_conservative_for_mixed_claims(self) -> None:
        extraction = evidence_assessment.SentenceClaimExtraction(
            sentence_index=0,
            sentence_text="Two claims.",
            context_text="Two claims.",
            verifiability="verifiable",
            claims=(claim(),),
        )
        assessments = [
            evidence_assessment.ClaimAssessment(
                claim_id="one", claim="One", sentence_index=0,
                evidence_status="supported", rationale="", citation_ids=("E1",),
            ),
            evidence_assessment.ClaimAssessment(
                claim_id="two", claim="Two", sentence_index=0,
                evidence_status="contradicted", rationale="", citation_ids=("E2",),
            ),
        ]
        self.assertEqual(
            evidence_assessment.sentence_evidence_status(extraction, assessments),
            "insufficient_evidence",
        )


if __name__ == "__main__":
    unittest.main()
