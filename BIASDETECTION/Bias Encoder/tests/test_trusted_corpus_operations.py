from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_ROOT / "LLM-inference"
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))

import corpus_operations
import evidence_assessment
import evidence_retrieval
import live_provider_checks
import retrieval_eval_runner
import retrieval_store
import runtime_config
import source_fetch
import trusted_source_registry


class FakeResponse:
    def __init__(self, payload: dict, headers: Message | None = None) -> None:
        self.body = json.dumps(payload).encode("utf-8")
        self.headers = headers or Message()
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, _limit: int = -1) -> bytes:
        return self.body


class SlowRetriever:
    def __init__(self, provider: str, delay: float) -> None:
        self.provider = provider
        self.delay = delay

    def retrieve(self, request):
        time.sleep(self.delay)
        return evidence_retrieval.EvidenceResult(
            status="none_found",
            items=[],
            provider=self.provider,
        )


class UnitEmbedder:
    @property
    def dimension(self):
        return 3

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class TrustedCorpusOperationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = trusted_source_registry.TrustedSourceRegistry.from_file(
            LLM_DIR / "trusted_sources.json"
        )

    def test_v2_registry_separates_fetch_ingest_and_sec_exclusions(self) -> None:
        bls = self.registry.evaluate_url("https://www.bls.gov/cpi/", purpose="fetch")
        ap_discovery = self.registry.evaluate_url(
            "https://apnews.com/article/example", purpose="discover"
        )
        ap_fetch = self.registry.evaluate_url(
            "https://apnews.com/article/example", purpose="fetch"
        )
        edgar = self.registry.evaluate_url(
            "https://www.sec.gov/Archives/edgar/data/123/report.htm", purpose="ingest"
        )

        self.assertTrue(bls.allowed)
        self.assertTrue(ap_discovery.allowed)
        self.assertFalse(ap_fetch.allowed)
        self.assertFalse(edgar.allowed)

    def test_explicit_env_file_does_not_override_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.env"
            path.write_text(
                "BRAVE_SEARCH_API_KEY=file-key\nQDRANT_URL=https://cluster.example\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "process-key"}, clear=True):
                runtime_config.load_explicit_env_file(path)
                self.assertEqual(os.environ["BRAVE_SEARCH_API_KEY"], "process-key")
                self.assertEqual(os.environ["QDRANT_URL"], "https://cluster.example")

    def test_ssrf_rejects_private_dns_answers_before_open(self) -> None:
        opener = mock.Mock()
        with self.assertRaises(source_fetch.SourceFetchError) as raised:
            source_fetch.fetch_source_document(
                "https://www.bls.gov/report",
                self.registry,
                contact="research@example.com",
                deadline=time.monotonic() + 1,
                opener=opener,
                resolver=lambda *_args, **_kwargs: [
                    (2, 1, 6, "", ("127.0.0.1", 443))
                ],
            )
        self.assertEqual(raised.exception.code, "unsafe_address")
        opener.open.assert_not_called()

    def test_trafilatura_extracts_main_text_and_tables(self) -> None:
        html = (
            "<html><head><title>Agency report</title>"
            "<link rel='canonical' href='https://www.bls.gov/report'></head>"
            "<body><nav>Navigation only</nav><main><h1>Agency report</h1><p>"
            + ("Official economic evidence describes the measured series in detail. " * 20)
            + "</p><table><tr><th>Year</th><th>Value</th></tr>"
            "<tr><td>2026</td><td>42</td></tr></table></main>"
            "<footer>Footer only</footer></body></html>"
        ).encode("utf-8")
        document = source_fetch.extract_source_document(
            html,
            content_type="text/html",
            charset="utf-8",
            source_url="https://www.bls.gov/report",
            registry=self.registry,
            fetched_at="2026-01-01T00:00:00Z",
        )
        self.assertEqual(document.canonical_url, "https://www.bls.gov/report")
        self.assertIn("Official economic evidence", document.text)
        self.assertIn("2026", document.text)
        self.assertNotIn("Navigation only", document.text)

    def test_blank_pdf_is_rejected_as_image_only_without_ocr(self) -> None:
        from pypdf import PdfWriter

        stream = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.write(stream)
        with self.assertRaises(source_fetch.SourceFetchError) as raised:
            source_fetch.extract_pdf_text(stream.getvalue())
        self.assertEqual(raised.exception.code, "image_only_pdf")

    def test_content_addressed_staging_is_idempotent(self) -> None:
        rule = self.registry.evaluate_url(
            "https://www.bls.gov/report", purpose="ingest"
        ).rule
        assert rule is not None
        document = source_fetch.ExtractedSourceDocument(
            source_url="https://www.bls.gov/report",
            canonical_url="https://www.bls.gov/report",
            title="Report",
            text="Reviewed public source text " * 20,
            content_type="text/plain",
            fetched_at="2026-01-01T00:00:00Z",
            extraction_version=source_fetch.TEXT_EXTRACTION_VERSION,
            archive_hash="a" * 64,
            content_hash=retrieval_store.sha256_text("Reviewed public source text " * 20),
        )
        with tempfile.TemporaryDirectory() as directory:
            first = source_fetch.stage_source_document(document, rule, staging_dir=directory)
            second = source_fetch.stage_source_document(document, rule, staging_dir=directory)
            self.assertEqual(first, second)
            self.assertEqual(len(list(Path(directory).glob("*.txt"))), 1)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)

    def test_initial_manifest_hash_and_dataset_shape_validate(self) -> None:
        manifest_path = next((LLM_DIR / "corpus_manifests").glob("initial-candidate-*.json"))
        manifest = corpus_operations.load_json_object(manifest_path)
        manifest_validation = corpus_operations.validate_manifest(
            manifest, registry=self.registry
        )
        dataset = retrieval_eval_runner.load_dataset(
            LLM_DIR / "retrieval_eval" / "trusted_retrieval_v1.json"
        )
        dataset_validation = retrieval_eval_runner.validate_dataset(dataset)

        self.assertTrue(manifest_validation.valid)
        self.assertEqual(manifest_validation.document_count, 40)
        self.assertEqual(manifest_validation.manifest_hash, manifest["manifest_hash"])
        self.assertTrue(dataset_validation.valid)
        self.assertFalse(dataset_validation.review_complete)
        self.assertEqual(dataset_validation.case_count, 120)

    def test_manifest_diff_detects_content_change(self) -> None:
        current = {"documents": [{"canonical_document_id": "doc-1", "content_hash": "a"}]}
        candidate = {"documents": [
            {"canonical_document_id": "doc-1", "content_hash": "b"},
            {"canonical_document_id": "doc-2", "content_hash": "c"},
        ]}
        diff = corpus_operations.diff_manifests(current, candidate)
        self.assertEqual(diff.added, ("doc-2",))
        self.assertEqual(diff.changed, ("doc-1",))

    def test_manifest_approval_writes_immutable_hash_and_verified_pointer(self) -> None:
        url = "https://www.bls.gov/report"
        document_id = retrieval_store.stable_document_id(url)
        manifest = {
            "schema_version": corpus_operations.MANIFEST_SCHEMA_VERSION,
            "manifest_id": "reviewed-test",
            "created_at": "2026-01-01T00:00:00Z",
            "policy_version": "trusted_sources/v2",
            "collection": retrieval_store.DEFAULT_COLLECTION,
            "embedding": {
                "model": retrieval_store.DEFAULT_EMBEDDING_MODEL,
                "distance": "cosine",
                "vector_name": retrieval_store.DEFAULT_VECTOR_NAME,
                "content_field": retrieval_store.DEFAULT_CONTENT_FIELD,
                "chunk_words": 350,
                "chunk_overlap_words": 50,
            },
            "review": {"status": "pending"},
            "documents": [{
                "source_id": "us_bls",
                "canonical_document_id": document_id,
                "canonical_url": url,
                "title": "Reviewed report",
                "published_at": "2025-12-31T00:00:00Z",
                "fetched_at": "2026-01-01T00:00:00Z",
                "content_hash": "a" * 64,
                "archive_hash": "b" * 64,
                "extraction_version": "trafilatura-main-text/v1",
                "etag": "etag",
                "last_modified": "Wed, 01 Jan 2026 00:00:00 GMT",
                "rights_basis": {"id": "us_bls_public_domain_text", "url": "https://www.bls.gov/opub/copyright-information.htm"},
                "rights_review": {
                    "status": "approved",
                    "reviewer": "reviewer@example.com",
                    "reviewed_at": "2026-01-02T00:00:00Z",
                    "third_party_marked": False,
                },
                "refresh_after": "2026-06-01T00:00:00Z",
                "retention_until": "2027-01-01T00:00:00Z",
                "status": "approved",
                "expected_point_count": 1,
            }],
        }
        manifest["manifest_hash"] = corpus_operations.manifest_hash(manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.json"
            candidate.write_text(json.dumps(manifest), encoding="utf-8")
            dry_run = corpus_operations.approve_manifest(
                candidate,
                registry=self.registry,
                reviewer="manifest-reviewer@example.com",
                manifest_dir=root / "manifests",
                apply=False,
            )
            self.assertEqual(dry_run["status"], "would_approve")
            self.assertFalse((root / "manifests").exists())
            applied = corpus_operations.approve_manifest(
                candidate,
                registry=self.registry,
                reviewer="manifest-reviewer@example.com",
                manifest_dir=root / "manifests",
                apply=True,
            )
            current = corpus_operations.load_current_manifest(root / "manifests")
            self.assertEqual(applied["status"], "approved")
            self.assertEqual(current["manifest_hash"], applied["manifest_hash"])

    def test_expiry_is_dry_run_and_apply_requires_verified_backup(self) -> None:
        manifest = {
            "review": {"status": "approved"},
            "documents": [{
                "canonical_document_id": "doc-expired",
                "retention_until": "2020-01-01T00:00:00Z",
            }],
        }
        store = SimpleNamespace(
            config=SimpleNamespace(collection="trusted_corpus_v1"),
            delete_documents=mock.Mock(return_value=2),
        )
        result = corpus_operations.apply_expiry(
            store, manifest, backup_dir=Path("missing"), apply=False
        )
        self.assertEqual(result["status"], "would_delete")
        store.delete_documents.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(corpus_operations.CorpusOperationError):
                corpus_operations.apply_expiry(
                    store, manifest, backup_dir=directory, apply=True
                )
        store.delete_documents.assert_not_called()

    def test_exact_statistics_uses_every_scroll_page(self) -> None:
        schema = retrieval_store.CorpusCollectionSchema(
            vector_size=3,
            embedding_model="test-embedder/v1",
            vector_name="content",
            content_field="chunk_text",
        )

        class PagedClient:
            def __init__(self):
                self.offsets = []
                self.records = [
                    SimpleNamespace(id=f"point-{index}", payload={
                        "document_id": f"doc-{index // 2}",
                        "source_id": "us_bls" if index < 2 else "us_gao",
                        "content_hash": f"hash-{index // 2}",
                        "retention_until": "2030-01-01T00:00:00Z",
                    })
                    for index in range(5)
                ]

            def collection_exists(self, _name):
                return True

            def get_collection(self, _name):
                return {
                    "vectors": {"content": {"size": 3, "distance": "Cosine"}},
                    "metadata": schema.collection_metadata,
                    "points_count": 5,
                }

            def scroll(self, *, offset=None, limit, **_kwargs):
                start = int(offset or 0)
                self.offsets.append(offset)
                page = self.records[start : start + 2]
                next_offset = start + len(page) if start + len(page) < len(self.records) else None
                return page, next_offset

        client = PagedClient()
        store = retrieval_store.QdrantCorpusStore(
            retrieval_store.QdrantCorpusConfig(
                local_path=":memory:", embedding_model="test-embedder/v1"
            ),
            client=client,
        )
        statistics = corpus_operations.exact_statistics(store)
        self.assertEqual(statistics["chunk_count"], 5)
        self.assertEqual(statistics["document_count"], 3)
        self.assertEqual(client.offsets, [None, 2, 4])

    def test_backup_writes_hashes_and_rotates_generations(self) -> None:
        class BackupStore:
            config = SimpleNamespace(collection="trusted_corpus_v1")

            def create_snapshot(self):
                return "snapshot-1.snapshot"

            def download_snapshot(self, _name, destination):
                path = Path(destination)
                path.write_bytes(b"verified snapshot bytes")
                return path

            def list_all_points(self):
                return ()

            def collection_info(self):
                return None

            def server_version(self):
                return "1.18.0"

            def list_snapshots(self):
                return (SimpleNamespace(name="snapshot-1.snapshot", creation_time="2026-01-01"),)

            def delete_snapshot(self, _name):
                raise AssertionError("only one snapshot should be retained")

        with tempfile.TemporaryDirectory() as directory:
            result = corpus_operations.create_verified_backup(
                BackupStore(),
                corpus_dir=directory,
                manifest_dir=Path(directory) / "manifests",
                apply=True,
            )
            self.assertEqual(result["status"], "backed_up")
            self.assertTrue(result["verified"])
            self.assertEqual(len(result["snapshot_sha256"]), 64)
            self.assertEqual(len(result["archive_sha256"]), 64)

    def test_brave_429_distinguishes_burst_retry_and_monthly_quota(self) -> None:
        burst_headers = Message()
        burst_headers["X-RateLimit-Remaining"] = "0, 100"
        burst_headers["X-RateLimit-Reset"] = "0, 2592000"
        burst_error = urllib.error.HTTPError(
            evidence_retrieval.BRAVE_SEARCH_URL, 429, "rate", burst_headers, None
        )
        success_headers = Message()
        success_headers["X-RateLimit-Remaining"] = "1, 99"
        success_headers["X-RateLimit-Reset"] = "1, 2592000"
        with mock.patch.object(
            evidence_retrieval.urllib.request,
            "urlopen",
            side_effect=[burst_error, FakeResponse({"web": {"results": []}}, success_headers)],
        ):
            results = evidence_retrieval.search_brave(
                evidence_retrieval.EvidenceQuery("site:bls.gov test", "test"),
                "secret",
                1,
                evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY,
            )
        self.assertEqual(results, [])
        self.assertEqual(results.diagnostics["attempts"], 2)

        quota_headers = Message()
        quota_headers["X-RateLimit-Remaining"] = "0, 0"
        quota_headers["X-RateLimit-Reset"] = "1, 2592000"
        quota_error = urllib.error.HTTPError(
            evidence_retrieval.BRAVE_SEARCH_URL, 429, "quota", quota_headers, None
        )
        with mock.patch.object(
            evidence_retrieval.urllib.request, "urlopen", side_effect=quota_error
        ):
            with self.assertRaises(evidence_retrieval.BraveTransportError) as raised:
                evidence_retrieval.search_brave(
                    evidence_retrieval.EvidenceQuery("test", "test"),
                    "secret",
                    1,
                    evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY,
                )
        self.assertEqual(raised.exception.code, "quota_exhausted")
        self.assertEqual(raised.exception.http_status, 429)

    def test_live_brave_nonce_hit_is_warning_but_other_checks_remain_hard(self) -> None:
        item = evidence_retrieval.EvidenceItem(
            title="Consumer Price Index",
            url="https://www.bls.gov/cpi/",
            domain="bls.gov",
            source_type="primary_official",
            published_date=None,
            snippet="Official CPI information",
            query="site:bls.gov Consumer Price Index",
            retrieval_reason="live_check",
        )
        trusted = evidence_retrieval.BraveSearchItems(
            [item], {"rate_limit": {"remaining": [1, 99]}, "attempts": 1}
        )
        nonce = evidence_retrieval.BraveSearchItems(
            [item], {"rate_limit": {"remaining": [1, 98]}, "attempts": 1}
        )
        with mock.patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}), mock.patch.object(
            evidence_retrieval, "search_brave", side_effect=[trusted, nonce]
        ):
            result = live_provider_checks.check_brave(LLM_DIR / "trusted_sources.json")
        self.assertEqual(result["authentication_status"], "passed")
        self.assertEqual(result["trusted_query_status"], "passed")
        self.assertEqual(result["nonce_status"], "nonempty_warning")
        self.assertEqual(result["warnings"][0]["code"], "nonce_nonempty_warning")

        missing_headers = evidence_retrieval.BraveSearchItems([item], {"attempts": 1})
        with mock.patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}), mock.patch.object(
            evidence_retrieval, "search_brave", return_value=missing_headers
        ):
            with self.assertRaises(live_provider_checks.LiveCheckError):
                live_provider_checks.check_brave(LLM_DIR / "trusted_sources.json")

    def test_qdrant_collection_listing_returns_only_redacted_names(self) -> None:
        client = SimpleNamespace(get_collections=mock.Mock(return_value=SimpleNamespace(
            collections=[SimpleNamespace(name="trusted_corpus_v1"), SimpleNamespace(name="other")]
        )))
        store = retrieval_store.QdrantCorpusStore(
            retrieval_store.QdrantCorpusConfig(local_path=":memory:"), client=client
        )
        self.assertEqual(store.list_collection_names(), ("other", "trusted_corpus_v1"))

    def test_source_excerpt_is_assessment_eligible_and_beats_snippet(self) -> None:
        snippet = evidence_retrieval.EvidenceItem(
            title="Discovery",
            url="https://www.bls.gov/report",
            domain="bls.gov",
            source_type="primary_official",
            published_date=None,
            snippet="Short snippet",
            query="claim",
            retrieval_reason="test",
            canonical_url="https://www.bls.gov/report",
            material_kind="search_snippet",
        )
        excerpt = evidence_retrieval.replace(
            snippet,
            title="Full source",
            snippet="Direct extracted text",
            material_kind="source_excerpt",
            provenance=evidence_retrieval.EvidenceProvenance(
                provider="source_fetch",
                query="claim",
                query_source="test",
                collection=None,
                document_id="doc-1",
                chunk_id="chunk-1",
                point_id=None,
                score=0.9,
            ),
        )
        deduped = evidence_retrieval._deduplicate_evidence_items([snippet, excerpt])
        citation = evidence_assessment.CitationRegistry().register(deduped)[0]
        self.assertEqual(deduped[0].material_kind, "source_excerpt")
        self.assertTrue(citation.assessment_eligible)

    def test_web_fetches_only_top_three_allowed_urls_and_emits_stable_excerpts(self) -> None:
        policy = evidence_retrieval.load_trusted_source_policy(
            LLM_DIR / "trusted_sources.json"
        )
        snippets = [
            evidence_retrieval.EvidenceItem(
                title=f"Result {index}",
                url=f"https://www.bls.gov/report-{index}",
                domain="bls.gov",
                source_type="primary_official",
                published_date=None,
                snippet="Discovery text",
                query="economic claim",
                retrieval_reason="selected_sentence",
            )
            for index in range(4)
        ]
        fetched_urls = []

        def fetcher(url, _registry, **_kwargs):
            fetched_urls.append(url)
            text = f"Direct official evidence for {url}. " * 20
            return source_fetch.ExtractedSourceDocument(
                source_url=url,
                canonical_url=url,
                title=f"Fetched {url}",
                text=text,
                content_type="text/html",
                fetched_at="2026-01-01T00:00:00Z",
                extraction_version=source_fetch.HTML_EXTRACTION_VERSION,
                archive_hash="a" * 64,
                content_hash=retrieval_store.sha256_text(text),
            )

        with tempfile.TemporaryDirectory() as directory:
            retriever = evidence_retrieval.BraveWebEvidenceRetriever(
                policy=policy,
                api_key="test-key",
                fetch_contact="review@example.com",
                source_fetcher=fetcher,
                source_embedding_provider=UnitEmbedder(),
                staging_dir=directory,
            )
            request = evidence_retrieval.EvidenceRequest(
                mode="web",
                claims=(evidence_retrieval.EvidenceClaim(
                    "claim-1", "economic claim", "economic context"
                ),),
                max_items=5,
                timeout_seconds=2,
            )
            with mock.patch.object(
                evidence_retrieval, "search_brave", return_value=snippets
            ):
                result = retriever.retrieve(request)
        excerpts = [item for item in result.items if item.material_kind == "source_excerpt"]
        self.assertEqual(len(fetched_urls), 3)
        self.assertEqual(len(excerpts), 3)
        self.assertTrue(all(item.provenance.provider == "source_fetch" for item in excerpts))
        self.assertTrue(all(item.provenance.document_id.startswith("doc_") for item in excerpts))
        self.assertTrue(all(item.provenance.chunk_id.startswith("chunk_") for item in excerpts))

    def test_hybrid_starts_both_providers_concurrently(self) -> None:
        retriever = evidence_retrieval.HybridEvidenceRetriever(
            policy=evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY,
            web_retriever=SlowRetriever("brave", 0.12),
            corpus_retriever=SlowRetriever("qdrant", 0.12),
        )
        request = evidence_retrieval.EvidenceRequest(
            mode="hybrid",
            claims=(evidence_retrieval.EvidenceClaim("claim", "test claim", "context"),),
            max_items=5,
            timeout_seconds=1,
        )
        started = time.perf_counter()
        result = retriever.retrieve(request)
        elapsed = time.perf_counter() - started
        self.assertEqual(result.status, "none_found")
        self.assertLess(elapsed, 0.21)


if __name__ == "__main__":
    unittest.main()
