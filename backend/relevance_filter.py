"""Relevance filter: keep only sentences that belong to the article's main content.

Runs BEFORE SBERT bias classification. Two conservative layers:

1. **Heuristics** — drop obvious non-article noise:
   - ``too_short``            very short UI-like fragments ("Read more", "Menu")
   - ``boilerplate_keyword``  newsletter/cookie/share/related/comment prompts
   - ``duplicate``            repeated sentences (nav items rendered twice, etc.)

2. **Semantic (optional)** — embed a topic anchor (title + lead paragraph) and
   each surviving sentence, then drop sentences whose cosine similarity to the
   anchor is below ``threshold`` — but ONLY if the sentence also looks UI-like
   (short, link-y, not long-form prose). Long-form prose is never removed
   semantically, so topically divergent but genuine article content (quotes,
   background, tangents) survives. Reason: ``low_topic_similarity``.

The embedding function is injected (``embed_fn``), so the caller decides which
model produces embeddings (currently the fine-tuned SBERT already in memory;
swappable for base all-mpnet-base-v2 later without touching this module).

Design rule: when unsure, KEEP the sentence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

LOGGER = logging.getLogger("bias_backend.relevance_filter")

DEFAULT_THRESHOLD = 0.18

# Lexical patterns that mark classic page furniture. Matched case-insensitively
# against the whole sentence; most only apply to short sentences (see below) so
# an article ABOUT cookies or newsletters is not censored.
BOILERPLATE_PATTERNS = [
    r"\bsign\s*up\b", r"\bsubscribe\b", r"\bnewsletter\b", r"\bclick here\b",
    r"\bread (?:more|next)\b", r"\brelated (?:articles?|stories|posts)\b",
    r"\brecommended for you\b", r"\btrending\b", r"\bmost (?:read|popular)\b",
    r"\badvertisement\b", r"\bsponsored\b", r"\bpromoted\b",
    r"\baccept (?:all )?cookies\b", r"\bcookie (?:policy|settings|preferences)\b",
    r"\bprivacy policy\b", r"\bterms of (?:service|use)\b",
    r"\ball rights reserved\b", r"\bfollow us\b", r"\bshare this\b",
    r"\bshare on\b", r"\blog ?in\b", r"\bsign ?in\b", r"\bcreate (?:an )?account\b",
    r"\bskip to (?:main )?content\b", r"\bdownload (?:the|our) app\b",
    r"\bwatch live\b", r"\bleave a comment\b", r"\bview all comments\b",
    r"\bloading\b", r"\bcontinue reading\b", r"\bback to top\b",
    r"\benable javascript\b", r"\bad[- ]?blocker\b", r"\bgetty images\b",
    r"^©", r"\bcopyright \d{4}\b",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# Patterns so unambiguous they may remove a sentence of any length.
STRONG_BOILERPLATE_RE = re.compile(
    r"(\baccept (?:all )?cookies\b|\ball rights reserved\b|^©|\bcopyright \d{4}\b|"
    r"\bskip to (?:main )?content\b|\benable javascript\b)",
    re.IGNORECASE,
)

WORD_RE = re.compile(r"[A-Za-z0-9''-]+")


@dataclass
class RemovedSentence:
    index: int          # position in the ORIGINAL sentence list
    text: str
    reason: str
    similarity: float | None = None

    def to_json(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "index": self.index,
            "text": self.text[:200],
            "reason": self.reason,
        }
        if self.similarity is not None:
            record["similarity"] = round(self.similarity, 4)
        return record


@dataclass
class RelevanceResult:
    kept_sentences: list[str]
    removed: list[RemovedSentence] = field(default_factory=list)
    anchor_text: str = ""
    semantic_ran: bool = False

    @property
    def removed_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.removed:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return dict(sorted(counts.items()))

    def to_meta(self, total_sentences: int) -> dict[str, Any]:
        return {
            "total_sentences": total_sentences,
            "sentences_after_relevance_filter": len(self.kept_sentences),
            "sentences_removed_as_irrelevant": len(self.removed),
            "removed_reason_counts": self.removed_reason_counts,
            "semantic_ran": self.semantic_ran,
        }


def split_into_paragraph_sentences(
    text: str, sentence_splitter: Callable[[str], list[str]]
) -> list[str]:
    """Split text into sentences while respecting paragraph/newline boundaries.

    The scraper sends article blocks separated by newlines; plain sentence
    splitters that normalize whitespace would glue unpunctuated UI fragments
    ("Menu") onto the next real sentence. Splitting per block prevents that,
    so the fragments stay isolated and the heuristics can remove them.
    """
    sentences: list[str] = []
    for block in re.split(r"\n+", text or ""):
        block = block.strip()
        if block:
            sentences.extend(sentence_splitter(block))
    return sentences


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def is_long_form_prose(text: str) -> bool:
    """Genuine article-style sentence: never removed by the semantic layer."""
    words = _word_count(text)
    if words >= 12 and len(text) >= 80:
        return True
    # Shorter sentences with sentence-final punctuation and a lowercase interior
    # (i.e. not Title Case menu items) also count as prose. UI fragments in the
    # wild rarely carry terminal punctuation; punctuated boilerplate ("Sign up
    # for our newsletter today.") is caught by the keyword lexicon before the
    # semantic layer ever sees it.
    if words >= 6 and text.rstrip().endswith((".", "!", "?", '"', "”")):
        interior = text[1:]
        return any(c.islower() for c in interior)
    return False


def looks_ui_like(text: str) -> bool:
    """Short, fragmentary, or link-cluster text — the only candidates the
    semantic layer is allowed to remove."""
    return not is_long_form_prose(text)


def apply_heuristics(sentences: list[str]) -> tuple[list[tuple[int, str]], list[RemovedSentence]]:
    """Layer 1. Returns (kept as (original_index, text), removed)."""
    kept: list[tuple[int, str]] = []
    removed: list[RemovedSentence] = []
    seen: set[str] = set()

    for index, sentence in enumerate(sentences):
        text = sentence.strip()
        words = _word_count(text)

        # Boilerplate keywords first, so short boilerplate gets the more
        # informative reason. Short sentences match the broad lexicon; only
        # unambiguous patterns may remove longer sentences, so an article that
        # merely *mentions* newsletters or cookies is kept.
        if STRONG_BOILERPLATE_RE.search(text) or (
            len(text) < 120 and words <= 20 and BOILERPLATE_RE.search(text)
        ):
            removed.append(RemovedSentence(index, text, "boilerplate_keyword"))
            continue

        # Very short UI fragments ("Menu", "Read more", "Home | News").
        if words <= 3 and len(text) < 30:
            removed.append(RemovedSentence(index, text, "too_short"))
            continue

        # Duplicates (identical after whitespace/case normalization).
        key = _normalize(text)
        if key in seen:
            removed.append(RemovedSentence(index, text, "duplicate"))
            continue
        seen.add(key)

        kept.append((index, text))

    return kept, removed


def _cosine_rows(anchor: Any, rows: Any) -> list[float]:
    """Cosine similarity of each row against the anchor vector.

    Works with torch tensors (the SBERT output) without importing torch here.
    """
    def norm(v: Any) -> Any:
        return (v * v).sum() ** 0.5

    anchor_norm = norm(anchor)
    sims: list[float] = []
    for i in range(rows.shape[0]):
        row = rows[i]
        denom = float(anchor_norm * norm(row))
        sims.append(float((anchor * row).sum()) / denom if denom > 0 else 0.0)
    return sims


def apply_semantic(
    kept: list[tuple[int, str]],
    anchor_text: str,
    embed_fn: Callable[[list[str]], Any],
    threshold: float,
) -> tuple[list[tuple[int, str]], list[RemovedSentence]]:
    """Layer 2. Removes low-similarity sentences that ALSO look UI-like."""
    if not kept or not anchor_text.strip():
        return kept, []

    # Only UI-like sentences are semantic-removal candidates; embedding prose
    # would be wasted work since it is exempt by design.
    candidate_positions = [pos for pos, (_, text) in enumerate(kept) if looks_ui_like(text)]
    if not candidate_positions:
        return kept, []

    texts = [anchor_text] + [kept[pos][1] for pos in candidate_positions]
    try:
        embeddings = embed_fn(texts)
    except Exception as exc:  # noqa: BLE001 — never let relevance kill analysis
        LOGGER.warning("Semantic relevance embedding failed; keeping all sentences: %s", exc)
        return kept, []

    sims = _cosine_rows(embeddings[0], embeddings[1:])

    removed: list[RemovedSentence] = []
    drop_positions: set[int] = set()
    for sim, pos in zip(sims, candidate_positions):
        if sim < threshold:
            index, text = kept[pos]
            removed.append(RemovedSentence(index, text, "low_topic_similarity", similarity=sim))
            drop_positions.add(pos)

    surviving = [pair for pos, pair in enumerate(kept) if pos not in drop_positions]
    return surviving, removed


def filter_sentences(
    sentences: list[str],
    title: str = "",
    lead_text: str = "",
    embed_fn: Callable[[list[str]], Any] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    enable_semantic: bool = True,
) -> RelevanceResult:
    """Run both layers. ``embed_fn`` None (or ``enable_semantic`` False) means
    heuristics only. Anchor prefers the scraper's lead_text, falling back to the
    title plus the first prose sentence."""
    kept, removed = apply_heuristics(sentences)

    anchor = (lead_text or "").strip()
    if not anchor:
        first_prose = next((t for _, t in kept if is_long_form_prose(t)), "")
        anchor = ". ".join(p for p in [(title or "").strip(), first_prose] if p)

    semantic_ran = False
    if enable_semantic and embed_fn is not None and anchor:
        kept, semantic_removed = apply_semantic(kept, anchor, embed_fn, threshold)
        removed.extend(semantic_removed)
        semantic_ran = True

    return RelevanceResult(
        kept_sentences=[text for _, text in kept],
        removed=sorted(removed, key=lambda r: r.index),
        anchor_text=anchor,
        semantic_ran=semantic_ran,
    )
