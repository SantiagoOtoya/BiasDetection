from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
for import_path in (PROJECT_ROOT, BACKEND_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import relevance_filter


class Vec:
    """Minimal vector supporting the ops relevance_filter._cosine_rows uses."""

    def __init__(self, values):
        self.values = [float(v) for v in values]

    def __mul__(self, other):
        return Vec(a * b for a, b in zip(self.values, other.values))

    def sum(self):
        return sum(self.values)


class Rows:
    """Minimal 2D embedding container (shape + indexing + slicing)."""

    def __init__(self, vecs):
        self.vecs = list(vecs)

    @property
    def shape(self):
        return (len(self.vecs), len(self.vecs[0].values) if self.vecs else 0)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Rows(self.vecs[key])
        return self.vecs[key]


ARTICLE_SENTENCE = (
    "The senate passed the controversial spending bill on Thursday after weeks "
    "of heated debate among lawmakers."
)
OPINION_SENTENCE = (
    "Critics called the measure a reckless giveaway that betrays hardworking "
    "families across the country."
)


class HeuristicTests(unittest.TestCase):
    def test_keeps_real_article_sentences(self) -> None:
        result = relevance_filter.filter_sentences(
            [ARTICLE_SENTENCE, OPINION_SENTENCE], title="Senate bill"
        )
        self.assertEqual(result.kept_sentences, [ARTICLE_SENTENCE, OPINION_SENTENCE])
        self.assertEqual(result.removed, [])

    def test_drops_too_short_ui_fragments(self) -> None:
        result = relevance_filter.filter_sentences(["Menu", ARTICLE_SENTENCE])
        self.assertEqual(result.kept_sentences, [ARTICLE_SENTENCE])
        self.assertEqual(result.removed[0].reason, "too_short")

    def test_drops_boilerplate_keywords(self) -> None:
        result = relevance_filter.filter_sentences(
            ["Sign up for our newsletter today.", ARTICLE_SENTENCE]
        )
        self.assertEqual(result.kept_sentences, [ARTICLE_SENTENCE])
        self.assertEqual(result.removed[0].reason, "boilerplate_keyword")

    def test_long_sentence_mentioning_newsletter_is_kept(self) -> None:
        long_mention = (
            "The company announced that its popular email newsletter, which "
            "reaches two million subscribers, will be discontinued next year "
            "as part of a broader restructuring effort."
        )
        result = relevance_filter.filter_sentences([long_mention])
        self.assertEqual(result.kept_sentences, [long_mention])

    def test_strong_boilerplate_removed_at_any_length(self) -> None:
        cookie_wall = (
            "Accept all cookies to continue enjoying personalized content, "
            "tailored advertising, improved site performance and social media "
            "features across all of our partner websites."
        )
        result = relevance_filter.filter_sentences([cookie_wall])
        self.assertEqual(result.kept_sentences, [])
        self.assertEqual(result.removed[0].reason, "boilerplate_keyword")

    def test_drops_duplicates_keeps_first(self) -> None:
        result = relevance_filter.filter_sentences(
            [ARTICLE_SENTENCE, ARTICLE_SENTENCE.upper()]
        )
        self.assertEqual(len(result.kept_sentences), 1)
        self.assertEqual(result.removed[0].reason, "duplicate")

    def test_removed_reason_counts(self) -> None:
        result = relevance_filter.filter_sentences(
            ["Menu", "Home", "Subscribe now!", ARTICLE_SENTENCE]
        )
        meta = result.to_meta(4)
        self.assertEqual(meta["total_sentences"], 4)
        self.assertEqual(meta["sentences_after_relevance_filter"], 1)
        self.assertEqual(meta["sentences_removed_as_irrelevant"], 3)
        self.assertEqual(meta["removed_reason_counts"]["too_short"], 2)


class SemanticTests(unittest.TestCase):
    @staticmethod
    def fake_embed(vectors_by_text: dict[str, list[float]]):
        def embed(texts: list[str]) -> Rows:
            return Rows(Vec(vectors_by_text.get(t, [1.0, 0.0])) for t in texts)
        return embed

    def test_ui_like_low_similarity_sentence_removed(self) -> None:
        off_topic_fragment = "Best air fryer deals this week"
        embed = self.fake_embed(
            {
                "anchor about the senate bill": [1.0, 0.0],
                off_topic_fragment: [0.0, 1.0],  # cosine 0.0 < threshold
            }
        )
        result = relevance_filter.filter_sentences(
            [ARTICLE_SENTENCE, off_topic_fragment],
            lead_text="anchor about the senate bill",
            embed_fn=embed,
            threshold=0.18,
        )
        self.assertEqual(result.kept_sentences, [ARTICLE_SENTENCE])
        self.assertEqual(result.removed[0].reason, "low_topic_similarity")
        self.assertTrue(result.semantic_ran)

    def test_long_form_prose_never_removed_semantically(self) -> None:
        divergent_prose = (
            "In a completely different development overseas, scientists reported "
            "an unexpected breakthrough in battery chemistry that could reshape "
            "the electric vehicle market within a decade."
        )
        embed = self.fake_embed(
            {
                "anchor about the senate bill": [1.0, 0.0],
                divergent_prose: [0.0, 1.0],
            }
        )
        result = relevance_filter.filter_sentences(
            [divergent_prose],
            lead_text="anchor about the senate bill",
            embed_fn=embed,
            threshold=0.18,
        )
        # Prose is exempt from semantic removal even at similarity 0.
        self.assertEqual(result.kept_sentences, [divergent_prose])

    def test_short_punctuated_factual_sentence_survives_semantic(self) -> None:
        short_factual = "Officials announced the plan on Monday afternoon."
        embed = self.fake_embed(
            {
                "anchor about the senate bill": [1.0, 0.0],
                short_factual: [0.0, 1.0],  # similarity 0 — still must be kept
            }
        )
        result = relevance_filter.filter_sentences(
            [short_factual],
            lead_text="anchor about the senate bill",
            embed_fn=embed,
            threshold=0.18,
        )
        self.assertEqual(result.kept_sentences, [short_factual])

    def test_embed_failure_keeps_all_sentences(self) -> None:
        def broken_embed(texts):
            raise RuntimeError("model unavailable")

        fragment = "Top winter jackets ranked by our editors"
        result = relevance_filter.filter_sentences(
            [ARTICLE_SENTENCE, fragment],
            lead_text="anchor about the senate bill",
            embed_fn=broken_embed,
        )
        self.assertIn(fragment, result.kept_sentences)

    def test_semantic_skipped_without_anchor(self) -> None:
        embed = self.fake_embed({})
        result = relevance_filter.filter_sentences(
            [ARTICLE_SENTENCE], lead_text="", title="", embed_fn=embed
        )
        # Anchor falls back to first prose sentence; semantic may run, but the
        # prose exemption keeps everything.
        self.assertEqual(result.kept_sentences, [ARTICLE_SENTENCE])


if __name__ == "__main__":
    unittest.main()
