from __future__ import annotations

import os
import sys
import unittest
from dataclasses import fields
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_ROOT / "LLM-inference"
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))

import evidence_retrieval
import retrieval_store


def trusted_item(
    *,
    url: str = "https://www.federalreserve.gov/releases/z1/dataviz/dfa/",
) -> evidence_retrieval.EvidenceItem:
    return evidence_retrieval.EvidenceItem(
        title="Federal Reserve data",
        url=url,
        domain="federalreserve.gov",
        source_type="primary_official",
        published_date=None,
        snippet="Distributional Financial Accounts data.",
        query="Federal Reserve wealth data",
        retrieval_reason="selected_sentence",
    )


def web_request(
    *claims: evidence_retrieval.EvidenceClaim,
) -> evidence_retrieval.EvidenceRequest:
    return evidence_retrieval.EvidenceRequest(
        mode="web",
        claims=claims
        or (
            evidence_retrieval.EvidenceClaim(
                claim_id="claim-1",
                text="Federal Reserve wealth data",
                context_text="The article discusses household wealth.",
                selected_sentence_indices=(4,),
            ),
        ),
        max_items=5,
        timeout_seconds=1,
        request_id="request-1",
        article_id="article-1",
    )


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.text_batches: list[list[str]] = []

    def embed(self, texts):
        self.text_batches.append(list(texts))
        return [[1.0, 0.0, 0.0] for _text in texts]


class FakeCorpusStore:
    def __init__(self, *responses) -> None:
        self.config = SimpleNamespace(
            collection="trusted_corpus_v1",
            content_field="chunk_text",
            embedding_model="test-embedder/v1",
        )
        self.responses = responses or ((),)
        self.calls: list[dict] = []

    def search_points(self, vector, *, limit: int):
        self.calls.append({"vector": list(vector), "limit": limit})
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return tuple(response)


class StaticRetriever:
    def __init__(self, result: evidence_retrieval.EvidenceResult) -> None:
        self.result = result
        self.requests: list[evidence_retrieval.EvidenceRequest] = []

    def retrieve(self, request: evidence_retrieval.EvidenceRequest) -> evidence_retrieval.EvidenceResult:
        self.requests.append(request)
        return self.result


def corpus_point(
    *,
    point_id: str = "point-1",
    document_id: str = "doc-1",
    chunk_id: str = "chunk-1",
    canonical_url: str = "https://www.bls.gov/news/example.htm",
    content_hash: str = "content-1",
    score: float = 0.9,
) -> retrieval_store.ScoredCorpusPoint:
    return retrieval_store.ScoredCorpusPoint(
        id=point_id,
        score=score,
        payload={
            "point_id": point_id,
            "document_id": document_id,
            "chunk_id": chunk_id,
            "canonical_url": canonical_url,
            "source_url": canonical_url,
            "source_domain": evidence_retrieval.normalize_domain(canonical_url),
            "source_tier": "primary_official",
            "title": f"Document {document_id}",
            "chunk_text": f"Evidence from {document_id}.",
            "content_hash": content_hash,
            "published_at": "2026-01-02T00:00:00Z",
        },
    )


def ranked_item(
    *,
    title: str,
    url: str,
    claim_id: str,
    provider: str,
    document_id: str | None = None,
    content_hash: str | None = None,
    score: float | None = None,
) -> evidence_retrieval.EvidenceItem:
    canonical_url = evidence_retrieval.canonicalize_web_url(url) or url
    return evidence_retrieval.EvidenceItem(
        title=title,
        url=url,
        domain=evidence_retrieval.normalize_domain(url),
        source_type="primary_official",
        published_date=None,
        snippet=f"Evidence for {title}.",
        query="test query",
        retrieval_reason="selected_sentence",
        claim_ids=(claim_id,),
        canonical_url=canonical_url,
        provenance=evidence_retrieval.EvidenceProvenance(
            provider=provider,
            query="test query",
            query_source="selected_sentence",
            collection="trusted_corpus_v1" if provider == "qdrant" else None,
            document_id=document_id,
            chunk_id="chunk-1" if provider == "qdrant" else None,
            point_id="point-1" if provider == "qdrant" else None,
            score=score,
        ),
        content_hash=content_hash,
    )


def provider_result(
    provider: str,
    items: list[evidence_retrieval.EvidenceItem],
    *,
    status: str = "found",
    failures: list[evidence_retrieval.EvidenceFailure] | None = None,
) -> evidence_retrieval.EvidenceResult:
    return evidence_retrieval.EvidenceResult(
        status=status,
        items=items,
        provider=provider,
        failures=failures or [],
    )


class EvidenceRetrievalTests(unittest.TestCase):
    def test_retrieval_contract_has_no_assessment_verdict_field(self) -> None:
        names = {field.name for field in fields(evidence_retrieval.EvidenceItem)}
        self.assertNotIn("supports_claim", names)
        self.assertNotIn("verdict", names)

    def test_claim_inputs_are_normalized_and_keep_report_indices(self) -> None:
        claim = evidence_retrieval.EvidenceClaim(
            claim_id=" claim-1 ",
            text="  This   selected\nclaim needs review.  ",
            context_text="  Nearby\tcontext. ",
            selected_sentence_indices=[2, 5],
        )

        self.assertEqual(claim.claim_id, "claim-1")
        self.assertEqual(claim.text, "This selected claim needs review.")
        self.assertEqual(claim.context_text, "Nearby context.")
        self.assertEqual(claim.selected_sentence_indices, (2, 5))

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
            "wire_service",
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
        self.assertEqual(result.failures[0].stage, "configuration")
        self.assertEqual(result.failures[0].exception_type, "RetrieverUnavailableError")

    def test_retrieve_evidence_filters_and_caps_mocked_results(self) -> None:
        item = trusted_item()

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

    def test_select_evidence_retriever_routes_off_and_brave(self) -> None:
        off_retriever = evidence_retrieval.select_evidence_retriever("off")
        web_retriever = evidence_retrieval.select_evidence_retriever(
            "web",
            provider="brave",
            api_key="token",
        )
        corpus_retriever = evidence_retrieval.select_evidence_retriever("corpus")
        hybrid_retriever = evidence_retrieval.select_evidence_retriever(
            "hybrid",
            provider="brave",
            api_key="token",
        )

        self.assertIsInstance(off_retriever, evidence_retrieval.DisabledEvidenceRetriever)
        self.assertIsInstance(web_retriever, evidence_retrieval.BraveWebEvidenceRetriever)
        self.assertIsInstance(corpus_retriever, evidence_retrieval.CorpusEvidenceRetriever)
        self.assertIsInstance(hybrid_retriever, evidence_retrieval.HybridEvidenceRetriever)
        with self.assertRaises(evidence_retrieval.RetrieverUnavailableError):
            evidence_retrieval.select_evidence_retriever("web", provider="unknown")

    def test_web_retriever_enforces_trusted_https_policy_after_provider_call(self) -> None:
        untrusted = trusted_item(url="https://example.com/story")
        insecure = trusted_item(url="http://www.cdc.gov/data")
        credentialed = trusted_item(url="https://user@www.cdc.gov/data")
        trusted = trusted_item()

        with mock.patch.object(
            evidence_retrieval,
            "search_brave",
            return_value=[untrusted, insecure, credentialed, trusted],
        ):
            result = evidence_retrieval.retrieve_evidence_request(
                web_request(),
                provider="brave",
                api_key="token",
            )

        self.assertEqual(result.status, "found")
        self.assertEqual([item.url for item in result.items], [trusted.url])
        self.assertEqual(result.items[0].source_type, "primary_official")

    def test_web_result_populates_claim_query_and_result_provenance(self) -> None:
        request = web_request(
            evidence_retrieval.EvidenceClaim(
                claim_id="claim-wealth",
                text="Federal Reserve data show household wealth concentration.",
                context_text="The article claims wealth is concentrated at the top.",
                selected_sentence_indices=(7,),
            )
        )
        with mock.patch.object(evidence_retrieval, "search_brave", return_value=[trusted_item()]):
            result = evidence_retrieval.retrieve_evidence_request(
                request,
                provider="brave",
                api_key="token",
            )

        self.assertEqual(result.status, "found")
        self.assertEqual(result.requested_mode, "web")
        self.assertEqual(result.effective_mode, "web")
        self.assertEqual(result.request_id, "request-1")
        self.assertEqual(result.article_id, "article-1")
        self.assertGreaterEqual(result.elapsed_ms or 0, 0)
        self.assertEqual(result.queries[0].claim_ids, ("claim-wealth",))
        item = result.items[0]
        self.assertEqual(item.claim_ids, ("claim-wealth",))
        self.assertEqual(item.canonical_url, trusted_item().url)
        self.assertIsNotNone(item.retrieved_at)
        self.assertIsNotNone(item.provenance)
        assert item.provenance is not None
        self.assertEqual(item.provenance.provider, "brave")
        self.assertEqual(item.provenance.query, result.queries[0].text)
        self.assertEqual(item.provenance.query_source, "selected_sentence")
        self.assertEqual(item.provenance.collection, None)
        self.assertEqual(result.items_json()[0]["provenance"]["provider"], "brave")

    def test_web_retriever_returns_none_found_for_empty_provider_results(self) -> None:
        with mock.patch.object(evidence_retrieval, "search_brave", return_value=[]):
            result = evidence_retrieval.retrieve_evidence_request(
                web_request(),
                provider="brave",
                api_key="token",
            )

        self.assertEqual(result.status, "none_found")
        self.assertEqual(result.items, [])
        self.assertEqual(result.failures, [])
        self.assertTrue(result.queries)

    def test_provider_error_returns_safe_failure_metadata(self) -> None:
        with mock.patch.object(
            evidence_retrieval,
            "search_brave",
            side_effect=OSError("network unavailable"),
        ):
            result = evidence_retrieval.retrieve_evidence_request(
                web_request(),
                provider="brave",
                api_key="token",
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.items, [])
        self.assertEqual(result.error, "Brave web search failed")
        self.assertEqual(result.failures[0].stage, "search")
        self.assertEqual(result.failures[0].exception_type, "OSError")
        self.assertNotIn("network unavailable", result.error or "")

    def test_provider_error_after_valid_item_is_partial(self) -> None:
        request = web_request(
            evidence_retrieval.EvidenceClaim(
                claim_id="claim-1",
                text="Federal Reserve wealth data",
                context_text="Household wealth context.",
            ),
            evidence_retrieval.EvidenceClaim(
                claim_id="claim-2",
                text="Treasury distribution data",
                context_text="Household wealth context.",
            ),
        )
        with mock.patch.object(
            evidence_retrieval,
            "search_brave",
            side_effect=[[trusted_item()], OSError("network unavailable")],
        ):
            result = evidence_retrieval.retrieve_evidence_request(
                request,
                provider="brave",
                api_key="token",
            )

        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.failures[0].query_source, "selected_sentence")

    def test_corpus_retriever_returns_diverse_ranked_evidence_with_provenance(self) -> None:
        first = corpus_point(point_id="point-a", document_id="doc-a", score=0.95)
        duplicate_chunk = corpus_point(
            point_id="point-a2",
            document_id="doc-a",
            chunk_id="chunk-a2",
            score=0.90,
        )
        second = corpus_point(
            point_id="point-b",
            document_id="doc-b",
            chunk_id="chunk-b",
            canonical_url="https://www.gao.gov/products/example",
            content_hash="content-b",
            score=0.80,
        )
        store = FakeCorpusStore([first, duplicate_chunk, second])
        embedding_provider = FakeEmbeddingProvider()
        retriever = evidence_retrieval.CorpusEvidenceRetriever(
            store=store,
            embedding_provider=embedding_provider,
        )
        request = replace(web_request(), mode="corpus", max_items=2)

        result = retriever.retrieve(request)

        self.assertEqual(result.status, "found")
        self.assertEqual(result.requested_mode, "corpus")
        self.assertEqual(result.effective_mode, "corpus")
        self.assertEqual(len(result.items), 2)
        self.assertEqual(
            [item.provenance.document_id for item in result.items if item.provenance],
            ["doc-a", "doc-b"],
        )
        self.assertEqual(store.calls[0]["limit"], 6)
        self.assertTrue(embedding_provider.text_batches)
        item = result.items[0]
        assert item.provenance is not None
        self.assertEqual(item.content_hash, "content-1")
        self.assertEqual(item.provenance.provider, "qdrant")
        self.assertEqual(item.provenance.collection, "trusted_corpus_v1")
        self.assertEqual(item.provenance.document_id, "doc-a")
        self.assertEqual(item.provenance.chunk_id, "chunk-1")
        self.assertEqual(item.provenance.point_id, "point-a")
        self.assertEqual(item.provenance.score, 0.95)

    def test_corpus_retriever_returns_safe_error_when_qdrant_is_unavailable(self) -> None:
        store = FakeCorpusStore(retrieval_store.QdrantUnavailableError("offline"))
        retriever = evidence_retrieval.CorpusEvidenceRetriever(
            store=store,
            embedding_provider=FakeEmbeddingProvider(),
        )

        result = retriever.retrieve(replace(web_request(), mode="corpus"))

        self.assertEqual(result.status, "error")
        self.assertEqual(result.items, [])
        self.assertEqual(result.error, "Qdrant corpus search failed")
        self.assertEqual(result.failures[0].provider, "qdrant")
        self.assertEqual(result.failures[0].exception_type, "QdrantUnavailableError")

    def test_hybrid_invokes_both_retrievers_and_applies_the_result_cap(self) -> None:
        web = StaticRetriever(
            provider_result(
                "brave",
                [
                    ranked_item(
                        title="Web result",
                        url="https://www.cdc.gov/data/web-result",
                        claim_id="web-claim",
                        provider="brave",
                    )
                ],
            )
        )
        corpus = StaticRetriever(
            provider_result(
                "qdrant",
                [
                    ranked_item(
                        title="Corpus result",
                        url="https://www.bls.gov/news/corpus-result.htm",
                        claim_id="corpus-claim",
                        provider="qdrant",
                        document_id="doc-corpus",
                        content_hash="hash-corpus",
                        score=0.91,
                    )
                ],
            )
        )
        retriever = evidence_retrieval.HybridEvidenceRetriever(
            policy=evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY,
            web_retriever=web,
            corpus_retriever=corpus,
        )

        result = retriever.retrieve(replace(web_request(), mode="hybrid", max_items=1))

        self.assertEqual(result.status, "found")
        self.assertEqual(result.requested_mode, "hybrid")
        self.assertEqual(result.effective_mode, "hybrid")
        self.assertEqual(result.provider, "hybrid")
        self.assertEqual(len(result.items), 1)
        self.assertEqual(web.requests[0].mode, "web")
        self.assertEqual(corpus.requests[0].mode, "corpus")

    def test_hybrid_dispatcher_invokes_web_and_corpus_retrievers(self) -> None:
        store = FakeCorpusStore(
            [
                corpus_point(
                    document_id="doc-dispatch",
                    canonical_url="https://www.bls.gov/news/dispatch.htm",
                    content_hash="hash-dispatch",
                )
            ]
        )
        with mock.patch.object(
            evidence_retrieval,
            "search_brave",
            return_value=[trusted_item()],
        ):
            result = evidence_retrieval.retrieve_evidence_request(
                replace(web_request(), mode="hybrid", max_items=2),
                provider="brave",
                api_key="token",
                corpus_store=store,
                corpus_embedding_provider=FakeEmbeddingProvider(),
            )

        self.assertEqual(result.status, "found")
        self.assertEqual(result.provider, "hybrid")
        self.assertEqual(result.effective_mode, "hybrid")
        self.assertEqual(len(result.items), 2)
        self.assertTrue(store.calls)
        self.assertEqual(
            {item.provenance.provider for item in result.items if item.provenance},
            {"brave", "qdrant"},
        )

    def test_hybrid_deduplicates_by_url_document_and_content_hash(self) -> None:
        fused = evidence_retrieval._fuse_evidence_rankings(
            [
                [
                    ranked_item(
                        title="URL duplicate (web)",
                        url="https://www.cdc.gov/data/same?utm_source=test",
                        claim_id="url-web",
                        provider="brave",
                    ),
                    ranked_item(
                        title="Document duplicate (web)",
                        url="https://www.cdc.gov/data/document-one",
                        claim_id="document-web",
                        provider="brave",
                        document_id="doc-shared",
                    ),
                    ranked_item(
                        title="Content duplicate (web)",
                        url="https://www.cdc.gov/data/content-one",
                        claim_id="content-web",
                        provider="brave",
                        content_hash="hash-shared",
                    ),
                ],
                [
                    ranked_item(
                        title="URL duplicate (corpus)",
                        url="https://www.cdc.gov/data/same",
                        claim_id="url-corpus",
                        provider="qdrant",
                        document_id="doc-url",
                        score=0.9,
                    ),
                    ranked_item(
                        title="Document duplicate (corpus)",
                        url="https://www.bls.gov/news/document-two.htm",
                        claim_id="document-corpus",
                        provider="qdrant",
                        document_id="doc-shared",
                        score=0.8,
                    ),
                    ranked_item(
                        title="Content duplicate (corpus)",
                        url="https://www.gao.gov/products/content-two",
                        claim_id="content-corpus",
                        provider="qdrant",
                        document_id="doc-content",
                        content_hash="hash-shared",
                        score=0.7,
                    ),
                    ranked_item(
                        title="Unique result",
                        url="https://www.nih.gov/data/unique",
                        claim_id="unique",
                        provider="qdrant",
                        document_id="doc-unique",
                        content_hash="hash-unique",
                        score=0.6,
                    ),
                ],
            ],
            max_items=10,
        )

        self.assertEqual(len(fused), 4)
        url_duplicate = next(item for item in fused if "url-web" in item.claim_ids)
        document_duplicate = next(item for item in fused if "document-web" in item.claim_ids)
        content_duplicate = next(item for item in fused if "content-web" in item.claim_ids)
        self.assertEqual(url_duplicate.claim_ids, ("url-web", "url-corpus"))
        self.assertEqual(document_duplicate.claim_ids, ("document-web", "document-corpus"))
        self.assertEqual(content_duplicate.claim_ids, ("content-web", "content-corpus"))

    def test_hybrid_rrf_ordering_is_deterministic_and_preserves_winner_provenance(self) -> None:
        web_ranking = [
            ranked_item(
                title="A",
                url="https://www.cdc.gov/data/a",
                claim_id="a",
                provider="brave",
            ),
            ranked_item(
                title="B web",
                url="https://www.cdc.gov/data/b",
                claim_id="b-web",
                provider="brave",
                document_id="doc-b",
            ),
        ]
        corpus_ranking = [
            ranked_item(
                title="B corpus",
                url="https://www.cdc.gov/data/b",
                claim_id="b-corpus",
                provider="qdrant",
                document_id="doc-b",
                score=0.99,
            ),
            ranked_item(
                title="C",
                url="https://www.bls.gov/news/c.htm",
                claim_id="c",
                provider="qdrant",
                document_id="doc-c",
                score=0.8,
            ),
        ]

        rankings = [
            [item.title for item in evidence_retrieval._fuse_evidence_rankings(
                [web_ranking, corpus_ranking], max_items=5
            )]
            for _ in range(3)
        ]
        fused = evidence_retrieval._fuse_evidence_rankings(
            [web_ranking, corpus_ranking], max_items=5
        )

        self.assertEqual(rankings, [["B corpus", "A", "C"]] * 3)
        self.assertEqual(fused[0].claim_ids, ("b-web", "b-corpus"))
        assert fused[0].provenance is not None
        self.assertEqual(fused[0].provenance.provider, "qdrant")
        self.assertEqual(fused[0].provenance.score, 0.99)

    def test_hybrid_returns_partial_results_when_one_provider_fails(self) -> None:
        web = StaticRetriever(
            provider_result(
                "brave",
                [
                    ranked_item(
                        title="Web fallback",
                        url="https://www.cdc.gov/data/fallback",
                        claim_id="web",
                        provider="brave",
                    )
                ],
            )
        )
        corpus = StaticRetriever(
            provider_result(
                "qdrant",
                [],
                status="error",
                failures=[
                    evidence_retrieval.EvidenceFailure(
                        provider="qdrant",
                        stage="search",
                        message="Qdrant corpus search failed",
                    )
                ],
            )
        )
        retriever = evidence_retrieval.HybridEvidenceRetriever(
            policy=evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY,
            web_retriever=web,
            corpus_retriever=corpus,
        )

        result = retriever.retrieve(replace(web_request(), mode="hybrid"))

        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.error, "One or more evidence providers failed")
        self.assertEqual(result.failures[0].provider, "qdrant")

    def test_hybrid_returns_partial_results_when_web_is_unavailable(self) -> None:
        web = StaticRetriever(
            provider_result(
                "brave",
                [],
                status="error",
                failures=[
                    evidence_retrieval.EvidenceFailure(
                        provider="brave",
                        stage="configuration",
                        message="Brave web search is unavailable",
                    )
                ],
            )
        )
        corpus = StaticRetriever(
            provider_result(
                "qdrant",
                [
                    ranked_item(
                        title="Corpus fallback",
                        url="https://www.bls.gov/news/fallback.htm",
                        claim_id="corpus",
                        provider="qdrant",
                        document_id="doc-fallback",
                        score=0.8,
                    )
                ],
            )
        )
        retriever = evidence_retrieval.HybridEvidenceRetriever(
            policy=evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY,
            web_retriever=web,
            corpus_retriever=corpus,
        )

        result = retriever.retrieve(replace(web_request(), mode="hybrid"))

        self.assertEqual(result.status, "partial")
        self.assertEqual([item.title for item in result.items], ["Corpus fallback"])
        self.assertEqual(result.failures[0].provider, "brave")

    def test_hybrid_returns_safe_error_when_both_providers_fail(self) -> None:
        web = StaticRetriever(provider_result("brave", [], status="error"))
        corpus = StaticRetriever(provider_result("qdrant", [], status="error"))
        retriever = evidence_retrieval.HybridEvidenceRetriever(
            policy=evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY,
            web_retriever=web,
            corpus_retriever=corpus,
        )

        result = retriever.retrieve(replace(web_request(), mode="hybrid"))

        self.assertEqual(result.status, "error")
        self.assertEqual(result.items, [])
        self.assertEqual(result.error, "One or more evidence providers failed")
        self.assertEqual([failure.provider for failure in result.failures], ["brave", "qdrant"])

    def test_off_mode_is_safe_and_does_not_call_brave(self) -> None:
        with mock.patch.object(evidence_retrieval, "search_brave") as search:
            result = evidence_retrieval.retrieve_evidence(
                selected_sentence_texts=["A factual claim that needs evidence."],
                context_text="A factual claim that needs evidence.",
                provider="brave",
                max_items=5,
                timeout_seconds=1,
                mode="off",
            )

        self.assertEqual(result.status, "not_requested")
        self.assertEqual(result.requested_mode, "off")
        self.assertEqual(result.effective_mode, "off")
        self.assertEqual(result.items, [])
        search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
