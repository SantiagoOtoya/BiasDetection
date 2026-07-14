from __future__ import annotations

import io
import sys
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_ROOT / "LLM-inference"
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))

import ingest_trusted_corpus
import retrieval_store


class FakeQdrantClient:
    """In-memory Qdrant-shaped fake used without qdrant-client installed."""

    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}
        self.points: dict[str, dict[str, retrieval_store.CorpusPoint]] = {}
        self.payload_indexes: list[tuple[str, str, str]] = []
        self.upsert_calls = 0
        self.delete_calls: list[tuple[str, tuple[str, ...]]] = []
        self.query_calls: list[dict] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, *, collection_name: str, vectors_config, metadata) -> None:
        self.collections[collection_name] = {
            "vectors": vectors_config,
            "metadata": dict(metadata),
        }
        self.points[collection_name] = {}

    def get_collection(self, collection_name: str):
        return self.collections[collection_name]

    def create_payload_index(
        self, *, collection_name: str, field_name: str, field_schema, wait: bool
    ) -> None:
        self.payload_indexes.append((collection_name, field_name, str(field_schema)))

    def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter=None,
        filter=None,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ):
        selected_filter = scroll_filter if scroll_filter is not None else filter
        records = [
            point
            for point in self.points[collection_name].values()
            if self._matches(point.payload, selected_filter)
        ]
        return records[:limit], None

    def upsert(self, *, collection_name: str, points, wait: bool) -> None:
        self.upsert_calls += 1
        for point in points:
            self.points[collection_name][point.id] = point

    def query_points(
        self,
        *,
        collection_name: str,
        query,
        using: str,
        query_filter=None,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ):
        self.query_calls.append(
            {
                "collection_name": collection_name,
                "query": list(query),
                "using": using,
                "query_filter": query_filter,
                "limit": limit,
            }
        )
        records = []
        for point in self.points[collection_name].values():
            if not self._matches(point.payload, query_filter):
                continue
            score = sum(float(left) * float(right) for left, right in zip(query, point.vector))
            records.append({"id": point.id, "payload": point.payload, "score": score})
        records.sort(key=lambda record: (-record["score"], record["id"]))
        return {"points": records[:limit]}

    def delete(self, *, collection_name: str, points_selector, wait: bool) -> None:
        point_ids = tuple(points_selector)
        self.delete_calls.append((collection_name, point_ids))
        for point_id in point_ids:
            self.points[collection_name].pop(point_id, None)

    @staticmethod
    def _matches(payload: dict, filter_spec) -> bool:
        if filter_spec is None:
            return True
        assert isinstance(filter_spec, retrieval_store.MetadataFilterSpec)
        for condition in filter_spec.conditions:
            value = payload.get(condition.field_name)
            if condition.any_of and str(value) not in condition.any_of:
                return False
            if condition.gte and (value is None or str(value) < condition.gte):
                return False
            if condition.lte and (value is None or str(value) > condition.lte):
                return False
        return True


def source_metadata(
    *,
    source_id: str = "us_bls",
    source_url: str = "https://www.bls.gov/news/example.htm",
    canonical_url: str | None = None,
    source_tier: str = "primary_official",
    license_id: str = "us_bls_public_domain_text",
) -> retrieval_store.SourceMetadata:
    return retrieval_store.SourceMetadata(
        source_id=source_id,
        source_name="Test public source",
        source_url=source_url,
        canonical_url=canonical_url or source_url,
        source_domain=retrieval_store.normalise_domain(canonical_url or source_url),
        source_tier=source_tier,
        source_policy_version="trusted_sources/v1",
        license_id=license_id,
        license_url="https://example.gov/license",
        redistribution_allowed=True,
        retention_policy="review_after_365_days",
        retention_days=365,
    )


def document(
    *,
    text: str = "One two three four five six",
    source: retrieval_store.SourceMetadata | None = None,
    published_at: str | None = "2026-01-02",
) -> retrieval_store.CorpusDocument:
    return retrieval_store.CorpusDocument(
        title="A public document",
        text=text,
        source=source or source_metadata(),
        published_at=published_at,
        fetched_at="2026-01-03T04:05:06Z",
    )


def vectors_for_document(
    value: retrieval_store.CorpusDocument,
    *,
    chunk_words: int = 3,
    chunk_overlap_words: int = 1,
) -> list[list[float]]:
    chunks = retrieval_store.chunk_document_text(
        value.text,
        chunk_words=chunk_words,
        chunk_overlap_words=chunk_overlap_words,
    )
    return [[float(index + 1), 0.0, 0.5] for index in range(len(chunks))]


class QdrantCorpusTests(unittest.TestCase):
    def make_store(self, client: FakeQdrantClient | None = None) -> tuple[retrieval_store.QdrantCorpusStore, FakeQdrantClient]:
        fake = client or FakeQdrantClient()
        config = retrieval_store.QdrantCorpusConfig(
            local_path=":memory:",
            embedding_model="test-embedder/v1",
        )
        return retrieval_store.QdrantCorpusStore(config, client=fake), fake

    def test_configuration_supports_local_path_and_remote_url_modes(self) -> None:
        local = retrieval_store.QdrantCorpusConfig.from_environment(
            {"QDRANT_PATH": ".qdrant-local"}
        )
        remote = retrieval_store.QdrantCorpusConfig.from_environment(
            {
                "QDRANT_URL": "https://example.qdrant.io",
                "QDRANT_API_KEY": "token",
            }
        )

        self.assertEqual(local.mode, "local")
        self.assertEqual(local.local_path, ".qdrant-local")
        self.assertEqual(remote.mode, "remote")
        self.assertEqual(remote.url, "https://example.qdrant.io")
        with self.assertRaises(retrieval_store.CorpusConfigurationError):
            retrieval_store.QdrantCorpusConfig(
                local_path=".qdrant-local", url="http://127.0.0.1:6333"
            )

    def test_collection_schema_is_created_validated_and_indexed(self) -> None:
        store, fake = self.make_store()
        schema = retrieval_store.CorpusCollectionSchema(
            vector_size=3,
            vector_name="content",
            content_field="chunk_text",
            embedding_model="test-embedder/v1",
        )

        self.assertTrue(store.ensure_collection(schema))
        self.assertFalse(store.ensure_collection(schema))
        collection = fake.collections["trusted_corpus_v1"]
        self.assertEqual(collection["vectors"]["content"]["size"], 3)
        self.assertEqual(collection["vectors"]["content"]["distance"], "Cosine")
        self.assertEqual(
            collection["metadata"]["schema_version"], retrieval_store.CORPUS_SCHEMA_VERSION
        )
        self.assertIn(
            ("trusted_corpus_v1", "content_hash", "keyword"), fake.payload_indexes
        )
        self.assertIn(
            ("trusted_corpus_v1", "published_at", "datetime"), fake.payload_indexes
        )

    def test_existing_incompatible_collection_is_rejected_without_upsert(self) -> None:
        store, fake = self.make_store()
        fake.collections["trusted_corpus_v1"] = {
            "vectors": {"content": {"size": 99, "distance": "Cosine"}},
            "metadata": {
                "schema_version": retrieval_store.CORPUS_SCHEMA_VERSION,
                "embedding_model": "test-embedder/v1",
                "vector_name": "content",
                "vector_distance": "cosine",
                "content_field": "chunk_text",
                "chunking_version": retrieval_store.CHUNKING_VERSION,
            },
        }
        fake.points["trusted_corpus_v1"] = {}
        value = document()
        result = store.ingest_document(
            value,
            vectors_for_document(value),
            chunk_words=3,
            chunk_overlap_words=1,
        )

        self.assertEqual(result.status, "invalid_schema")
        self.assertEqual(fake.upsert_calls, 0)
        self.assertIn("incompatible", result.reason or "")

    def test_point_construction_stores_chunk_and_source_metadata(self) -> None:
        value = document(text="One two three four five")
        batch = retrieval_store.build_corpus_points(
            value,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            chunk_words=3,
            chunk_overlap_words=1,
        )

        self.assertEqual(len(batch.points), 2)
        payload = batch.points[0].payload
        self.assertEqual(payload["chunk_text"], "One two three")
        self.assertEqual(payload["source_tier"], "primary_official")
        self.assertEqual(payload["source_domain"], "bls.gov")
        self.assertEqual(payload["content_hash"], retrieval_store.sha256_text(value.text))
        self.assertEqual(payload["published_at"], "2026-01-02T00:00:00Z")
        self.assertEqual(payload["fetched_at"], "2026-01-03T04:05:06Z")
        self.assertEqual(payload["license_id"], "us_bls_public_domain_text")
        self.assertEqual(payload["retention_policy"], "review_after_365_days")
        self.assertTrue(payload["redistribution_allowed"])
        self.assertEqual(str(uuid.UUID(batch.points[0].id)), batch.points[0].id)

    def test_identifiers_are_deterministic_and_content_sensitive(self) -> None:
        value = document()
        vectors = vectors_for_document(value)
        first = retrieval_store.build_corpus_points(
            value, vectors, chunk_words=3, chunk_overlap_words=1
        )
        second = retrieval_store.build_corpus_points(
            value, vectors, chunk_words=3, chunk_overlap_words=1
        )
        changed = document(text="Nine two three four five six")
        changed_batch = retrieval_store.build_corpus_points(
            changed,
            vectors_for_document(changed),
            chunk_words=3,
            chunk_overlap_words=1,
        )

        self.assertEqual(first.document_id, second.document_id)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual([point.id for point in first.points], [point.id for point in second.points])
        self.assertNotEqual(first.content_hash, changed_batch.content_hash)
        self.assertNotEqual(first.points[0].id, changed_batch.points[0].id)

    def test_duplicate_ingestion_is_idempotent_and_cross_url_content_is_deduped(self) -> None:
        store, fake = self.make_store()
        first_document = document()
        first_vectors = vectors_for_document(first_document)
        first = store.ingest_document(
            first_document, first_vectors, chunk_words=3, chunk_overlap_words=1
        )
        repeat = store.ingest_document(
            first_document, first_vectors, chunk_words=3, chunk_overlap_words=1
        )
        copied_document = document(
            source=source_metadata(
                source_id="us_gao",
                source_url="https://www.gao.gov/products/example",
                source_tier="primary_official",
                license_id="us_gao_public_domain_text",
            )
        )
        copied = store.ingest_document(
            copied_document,
            vectors_for_document(copied_document),
            chunk_words=3,
            chunk_overlap_words=1,
        )

        self.assertEqual(first.status, "ingested")
        self.assertEqual(repeat.status, "duplicate")
        self.assertEqual(copied.status, "duplicate")
        self.assertEqual(copied.duplicate_document_id, first.document_id)
        self.assertEqual(fake.upsert_calls, 1)

    def test_changed_document_replaces_stale_points_without_losing_new_points(self) -> None:
        store, fake = self.make_store()
        original = document(text="One two three four five six seven eight")
        original_result = store.ingest_document(
            original,
            vectors_for_document(original),
            chunk_words=3,
            chunk_overlap_words=1,
        )
        changed = document(text="Nine ten eleven")
        changed_result = store.ingest_document(
            changed,
            vectors_for_document(changed),
            chunk_words=3,
            chunk_overlap_words=1,
        )

        self.assertEqual(original_result.status, "ingested")
        self.assertEqual(changed_result.status, "replaced")
        self.assertEqual(fake.upsert_calls, 2)
        self.assertTrue(fake.delete_calls)
        remaining = list(fake.points["trusted_corpus_v1"].values())
        self.assertEqual(len(remaining), changed_result.point_count)
        self.assertTrue(
            all(point.payload["content_hash"] == changed_result.content_hash for point in remaining)
        )

    def test_metadata_filters_select_by_source_license_and_dates(self) -> None:
        store, _ = self.make_store()
        bls = document(published_at="2026-01-02")
        gao = document(
            text="A different authoritative document",
            source=source_metadata(
                source_id="us_gao",
                source_url="https://www.gao.gov/products/example",
                license_id="us_gao_public_domain_text",
            ),
            published_at="2025-01-02",
        )
        store.ingest_document(bls, vectors_for_document(bls), chunk_words=3, chunk_overlap_words=1)
        store.ingest_document(gao, vectors_for_document(gao), chunk_words=3, chunk_overlap_words=1)

        filtered = store.list_points(
            retrieval_store.CorpusMetadataFilter(
                source_ids=("us_bls",),
                license_ids=("us_bls_public_domain_text",),
                published_from="2026-01-01",
                published_to="2026-12-31",
            )
        )
        self.assertTrue(filtered)
        self.assertTrue(all(point.payload["source_id"] == "us_bls" for point in filtered))

        filter_spec = retrieval_store.build_metadata_filter(
            retrieval_store.CorpusMetadataFilter(source_tiers=("primary_official",))
        )
        self.assertEqual(filter_spec.conditions[0].field_name, "source_tier")
        self.assertEqual(filter_spec.conditions[0].any_of, ("primary_official",))

    def test_dense_search_returns_ranked_payloads_without_mutating_collection(self) -> None:
        store, fake = self.make_store()
        first = document(text="One two three")
        second = document(
            text="Four five six",
            source=source_metadata(
                source_id="us_gao",
                source_url="https://www.gao.gov/products/example",
                license_id="us_gao_public_domain_text",
            ),
        )
        store.ingest_document(first, [[1.0, 0.0, 0.0]], chunk_words=3, chunk_overlap_words=1)
        store.ingest_document(second, [[0.5, 1.0, 0.0]], chunk_words=3, chunk_overlap_words=1)
        original_upserts = fake.upsert_calls

        hits = store.search_points([1.0, 0.0, 0.0], limit=5)

        self.assertEqual(len(hits), 2)
        self.assertGreaterEqual(hits[0].score, hits[1].score)
        self.assertEqual(hits[0].payload["canonical_url"], first.source.canonical_url)
        self.assertEqual(fake.query_calls[0]["using"], "content")
        self.assertEqual(fake.upsert_calls, original_upserts)

    def test_unavailable_qdrant_is_a_safe_result(self) -> None:
        class OfflineServer:
            def get_collections(self):
                raise OSError("server unavailable")

            def collection_exists(self, _collection_name: str) -> bool:
                raise OSError("server unavailable")

        config = retrieval_store.QdrantCorpusConfig(
            url="http://127.0.0.1:6333", embedding_model="test-embedder/v1"
        )
        store = retrieval_store.QdrantCorpusStore(config, client=OfflineServer())
        value = document()
        result = store.ingest_document(
            value,
            vectors_for_document(value),
            chunk_words=3,
            chunk_overlap_words=1,
        )

        self.assertEqual(store.health().status, "unavailable")
        self.assertEqual(result.status, "unavailable")
        self.assertNotIn("server unavailable", result.reason or "")

    def test_missing_optional_client_is_a_safe_unavailable_result(self) -> None:
        config = retrieval_store.QdrantCorpusConfig(
            local_path=":memory:", embedding_model="test-embedder/v1"
        )
        value = document()
        with mock.patch.dict(sys.modules, {"qdrant_client": None}):
            store = retrieval_store.QdrantCorpusStore(config)
            health = store.health()
            result = store.ingest_document(
                value,
                vectors_for_document(value),
                chunk_words=3,
                chunk_overlap_words=1,
            )

        self.assertEqual(health.status, "unavailable")
        self.assertEqual(result.status, "unavailable")
        self.assertIn("qdrant-client", result.reason or "")

    def test_source_policy_rejects_restricted_and_unknown_sources_before_fetch(self) -> None:
        registry = ingest_trusted_corpus.TrustedSourceRegistry.from_file(
            LLM_DIR / "trusted_sources.json"
        )
        self.assertTrue(
            registry.evaluate_url("https://www.bls.gov/news.release/example.htm").allowed
        )
        self.assertFalse(registry.evaluate_url("https://www.reuters.com/world/example").allowed)
        self.assertFalse(registry.evaluate_url("https://example.com/story").allowed)
        self.assertFalse(registry.evaluate_url("http://www.bls.gov/news.release/example.htm").allowed)

        with mock.patch.object(ingest_trusted_corpus, "fetch_remote_content") as fetch:
            stream = io.StringIO()
            with redirect_stdout(stream):
                exit_code = ingest_trusted_corpus.main(
                    ["--url", "https://www.reuters.com/world/example"]
                )
        self.assertEqual(exit_code, 2)
        fetch.assert_not_called()
        self.assertIn("rejected_policy", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
