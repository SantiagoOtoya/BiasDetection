"""Real analysis pipeline: SBERT classification + Llama report + fact-checker.

Reuses the existing, unmodified helpers from the repo-root ``infer_bias_llm.py``
(which itself imports ``finetune_all_mpnet_babe.py`` and ``evidence_retrieval.py``)
for:

  - sentence splitting
  - loading the fine-tuned SBERT + classification heads
  - calibrated / argmax sentence selection (with abstention)
  - per-sentence bias/opinion classification
  - building +/-2 sentence context windows
  - the media-bias system prompt and the Llama generator
  - trusted evidence retrieval

New backend code adds: assembling the checkpoint from the HF repo, a single
cohesive report prompt, the 0-100 bias score, and the conservative fact-checker.

Modes:
  - "gpu":         load Llama for real report + fact-check generation (needs CUDA).
  - "cpu":         same, but allow slow CPU Llama generation.
  - "prompt-only": SBERT + scoring + fact-check fallback, no Llama (fast local test).

Selection: `selection_mode` defaults to "auto", which uses a `calibration.json`
in the SBERT model dir when present and otherwise falls back to legacy argmax.

Evidence: disabled by default. When enabled it requires ``BRAVE_SEARCH_API_KEY``
and only accepts strictly trusted sources (see ``evidence_retrieval.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import fact_checker
import relevance_filter
import scoring
from model_loader import resolve_checkpoint
from pipeline import Analyzer
from schemas import AnalyzeRequest, AnalyzeResponse, SelectedSentence

LOGGER = logging.getLogger("bias_backend.real_pipeline")

# This module is the **v2** analyzer. The v2 inference code lives at the repo
# root; the v3 production stack lives under BIASDETECTION/ and is served by
# backend/v3_adapter.py. Exactly one stack may be active per process, enforced
# by stack_loader (module names collide between stacks).
import stack_loader

_STACK = stack_loader.load_stack("v2")
infer = _STACK.infer
evidence_retrieval = _STACK.evidence_retrieval


class RealAnalyzer(Analyzer):
    def __init__(
        self,
        mode: str = "gpu",
        context_sentences: int = 2,
        max_windows: int = 0,
        max_new_tokens: int = 700,
        temperature: float = 0.2,
        top_p: float = 0.9,
        assembled_dir: Path | None = None,
        selection_mode: str = "auto",
        enable_evidence: bool = False,
        evidence_provider: str = "brave",
        max_evidence_items: int = 5,
        evidence_timeout_seconds: float = 10.0,
        enable_relevance: bool = True,
        enable_semantic_relevance: bool = True,
        relevance_threshold: float = relevance_filter.DEFAULT_THRESHOLD,
    ) -> None:
        self.mode = mode
        self.context_sentences = context_sentences
        self.max_windows = max_windows
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.selection_mode = selection_mode
        self.enable_evidence = enable_evidence
        self.evidence_provider = evidence_provider
        self.max_evidence_items = max_evidence_items
        self.evidence_timeout_seconds = evidence_timeout_seconds
        self.enable_relevance = enable_relevance
        self.enable_semantic_relevance = enable_semantic_relevance
        self.relevance_threshold = relevance_threshold

        infer.training.import_training_dependencies()
        torch = infer.training.torch
        assert torch is not None
        self._torch = torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("SBERT device: %s", self.device)

        self.model_dir, heads_path = resolve_checkpoint(assembled_dir=assembled_dir)
        (
            self.model,
            self.bias_head,
            self.opinion_head,
            self.bias_label2id,
            self.opinion_label2id,
        ) = infer.load_sbert_and_heads(self.model_dir, heads_path, self.device)
        self.sbert_loaded = True
        LOGGER.info("SBERT + classification heads loaded.")

        # Resolve calibrated vs argmax selection. `auto` uses calibration.json in
        # the model dir when present, else falls back to legacy argmax.
        self.selection_config = self._resolve_selection_config()
        LOGGER.info(
            "Selection mode requested=%s effective=%s (%s).",
            self.selection_config.requested_mode,
            self.selection_config.effective_mode,
            self.selection_config.calibration_reason,
        )

        if self.enable_evidence:
            self.evidence_policy = evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY
            LOGGER.info(
                "Evidence retrieval ENABLED (provider=%s). Requires BRAVE_SEARCH_API_KEY.",
                self.evidence_provider,
            )
        else:
            self.evidence_policy = None
            LOGGER.info("Evidence retrieval disabled.")

        self.llm = None
        if mode in ("gpu", "cpu"):
            self.llm = infer.LlamaGenerator(
                model_name=infer.DEFAULT_LLM_MODEL,
                hf_token=None,
                allow_cpu=(mode == "cpu"),
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            self.llm_loaded = True
            LOGGER.info("Llama model loaded (%s).", infer.DEFAULT_LLM_MODEL)
        else:
            LOGGER.info("Prompt-only mode: skipping Llama load.")

    def _embed_texts(self, texts: list[str]):
        """Sentence embeddings via the already-loaded fine-tuned SBERT.

        Injected into the relevance filter so the embedding model stays
        swappable (e.g. base all-mpnet-base-v2 later) without changing the
        filter itself.
        """
        torch = self._torch
        with torch.no_grad():
            return infer.training.sentence_embeddings(
                self.model, texts, self.device
            ).detach().cpu()

    def _resolve_selection_config(self):
        selection_args = SimpleNamespace(
            selection_mode=self.selection_mode,
            calibration_file=None,  # defaults to <model_dir>/calibration.json
            sbert_model_dir=self.model_dir,
            bias_threshold=None,
            opinion_threshold=None,
        )
        return infer.resolve_selection_config(
            selection_args, self.bias_label2id, self.opinion_label2id
        )

    # --- analysis ---

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        # Paragraph-aware splitting: keeps unpunctuated UI fragments from being
        # glued onto real sentences before the relevance filter sees them.
        raw_sentences = relevance_filter.split_into_paragraph_sentences(
            request.text, infer.split_sentences
        )
        if not raw_sentences:
            return AnalyzeResponse(
                overall_score=0.0,
                score_caption="No readable sentences were found.",
                report="No readable article text was found on this page.",
                meta={"mode": self.mode, "sentence_count": 0, "selected_count": 0},
            )

        # --- relevance filter (before any bias classification) ---
        if self.enable_relevance:
            relevance = relevance_filter.filter_sentences(
                raw_sentences,
                title=request.title or "",
                lead_text=request.lead_text or "",
                embed_fn=self._embed_texts if self.enable_semantic_relevance else None,
                threshold=self.relevance_threshold,
                enable_semantic=self.enable_semantic_relevance,
            )
            sentences = relevance.kept_sentences
            relevance_meta = relevance.to_meta(len(raw_sentences))
        else:
            sentences = raw_sentences
            relevance_meta = {
                "total_sentences": len(raw_sentences),
                "sentences_after_relevance_filter": len(raw_sentences),
                "sentences_removed_as_irrelevant": 0,
                "removed_reason_counts": {},
                "semantic_ran": False,
                "disabled": True,
            }

        if not sentences:
            return AnalyzeResponse(
                overall_score=0.0,
                score_caption="No relevant article sentences were found.",
                report=(
                    "All extracted text looked like page furniture (menus, "
                    "prompts, footers) rather than article content."
                ),
                meta={
                    "mode": self.mode,
                    "sentence_count": 0,
                    "selected_count": 0,
                    "relevance": relevance_meta,
                },
            )

        predictions = infer.classify_sentences(
            sentences=sentences,
            model=self.model,
            bias_head=self.bias_head,
            opinion_head=self.opinion_head,
            bias_label2id=self.bias_label2id,
            opinion_label2id=self.opinion_label2id,
            device=self.device,
            batch_size=16,
            selection_config=self.selection_config,
        )

        selected = [p for p in predictions if p.selected]
        selected_indices = [p.index for p in selected]
        max_windows = request.max_windows or self.max_windows
        windows = infer.build_context_windows(
            sentences=sentences,
            selected_indices=selected_indices,
            context_sentences=self.context_sentences,
            max_windows=max_windows,
        )
        predictions_by_index = {p.index: p for p in predictions}

        # --- score ---
        signals = [
            scoring.sentence_signal(p.bias_probabilities, p.opinion_probabilities)
            for p in predictions
        ]
        score = scoring.compute_score(signals)
        level = scoring.bucket(score)
        caption = scoring.caption(score, len(selected), len(sentences))

        # --- trusted evidence (optional) ---
        evidence_items, evidence_status, evidence_error = self._retrieve_evidence(
            windows, predictions_by_index
        )

        # --- report ---
        report = self._build_report(
            windows, predictions_by_index, selected, evidence_items
        )

        # --- fact-check (reconciled with evidence) ---
        reviewed_sentences = [p.text for p in selected]
        context_text = "\n\n".join(w.text for w in windows)
        if self.enable_evidence:
            fact_checks = fact_checker.check_claims_with_evidence(
                llm=self.llm,
                reviewed_sentences=reviewed_sentences,
                context_text=context_text,
                evidence_items=evidence_items,
            )
        else:
            fact_checks = fact_checker.check_claims(
                llm=self.llm,
                reviewed_sentences=reviewed_sentences,
                context_text=context_text,
            )

        selected_out = [
            SelectedSentence(
                index=p.index,
                text=p.text,
                bias_label=p.bias_label,
                bias_probability=p.bias_probability,
                opinion_label=p.opinion_label,
                opinion_probability=p.opinion_probability,
                selection_reasons=p.selection_reasons or [],
                abstention_reasons=p.abstention_reasons or [],
            )
            for p in selected
        ]

        meta = {
            "mode": self.mode,
            # Relevant article sentences only (post-filter); totals live under
            # meta["relevance"].
            "sentence_count": len(sentences),
            "selected_count": len(selected),
            "window_count": len(windows),
            "llm_used": self.llm is not None,
            "relevance": relevance_meta,
            # Calibrated-selection provenance / debug metadata.
            "selection_policy": infer.selection_policy_to_json(self.selection_config),
            "abstention_summary": infer.build_abstention_summary(predictions),
            # Evidence provenance.
            "evidence_enabled": self.enable_evidence,
            "evidence_status": evidence_status,
            "evidence_item_count": len(evidence_items),
        }
        if evidence_error:
            meta["evidence_error"] = evidence_error

        return AnalyzeResponse(
            overall_score=score,
            bias_level=level,
            score_caption=caption,
            report=report,
            fact_checks=fact_checks,
            selected_sentences=selected_out,
            meta=meta,
        )

    # --- evidence ---

    def _retrieve_evidence(self, windows, predictions_by_index):
        """Retrieve trusted evidence across all windows.

        Returns ``(items, status, error)``. Status is one of the
        evidence_retrieval statuses, or "not_requested" when disabled.
        """
        if not self.enable_evidence or not windows:
            return [], "not_requested", None

        items: list = []
        seen_urls: set[str] = set()
        error: str | None = None
        found_any = False

        for window in windows:
            selected_texts = [
                predictions_by_index[index].text
                for index in window.selected_sentence_indices
                if index in predictions_by_index
            ]
            result = evidence_retrieval.retrieve_evidence(
                selected_sentence_texts=selected_texts,
                context_text=window.text,
                provider=self.evidence_provider,
                max_items=self.max_evidence_items,
                timeout_seconds=self.evidence_timeout_seconds,
                policy=self.evidence_policy
                or evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY,
            )
            if result.status == "error" and error is None:
                error = result.error
            if result.status == "found":
                found_any = True
            for item in result.items:
                if item.url in seen_urls:
                    continue
                seen_urls.add(item.url)
                items.append(item)
                if len(items) >= self.max_evidence_items:
                    break
            if len(items) >= self.max_evidence_items:
                break

        if items:
            status = "found"
        elif error is not None:
            status = "error"
        else:
            status = "none_found"
        _ = found_any
        return items, status, error

    # --- report helpers ---

    def _build_report(self, windows, predictions_by_index, selected, evidence_items) -> str:
        if not selected:
            return (
                "No strongly biased or opinionated wording was flagged in this "
                "page. The passage reads as largely factual based on the text provided."
            )
        if self.llm is None:
            return self._fallback_report(selected)

        import llm_utils

        prompt = self._build_combined_prompt(windows, predictions_by_index, evidence_items)
        try:
            return llm_utils.generate(
                self.llm,
                infer.MEDIA_BIAS_SYSTEM_PROMPT,
                prompt,
                max_new_tokens=self.max_new_tokens,
            ).strip()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Report generation failed: %s", exc)
            return self._fallback_report(selected)

    def _build_combined_prompt(self, windows, predictions_by_index, evidence_items) -> str:
        """One prompt covering all reviewed sentences + context, in article order.

        Produces a single cohesive report instead of one report per window. When
        trusted evidence is supplied, it is appended in the same shape the
        media-bias system prompt expects (citation ids like [E1]).
        """
        reviewed: list[str] = []
        for window in windows:
            for index in window.selected_sentence_indices:
                prediction = predictions_by_index.get(index)
                if prediction is not None and prediction.text.strip():
                    reviewed.append(prediction.text.strip())

        reviewed_json = json.dumps(
            [{"text": t} for t in reviewed], indent=2, ensure_ascii=False
        )
        context_block = "\n\n".join(w.text for w in windows)
        base_prompt = (
            "Analyze only the article passage below. The sentences listed under "
            "sentences_for_review are the article sentences that need focused "
            "analysis. Use surrounding_article_context only to interpret those "
            "sentences in their original chronological order.\n\n"
            f"sentences_for_review:\n{reviewed_json}\n\n"
            f"surrounding_article_context:\n{context_block}"
        )

        if not evidence_items:
            return base_prompt

        evidence_json = json.dumps(
            infer.evidence_items_for_prompt(evidence_items), indent=2, ensure_ascii=False
        )
        return (
            f"{base_prompt}\n\n"
            "trusted_external_evidence:\n"
            f"{evidence_json}\n\n"
            "Use trusted_external_evidence only to assess whether factual claims in "
            "the article passage are supported, contradicted, or not established by "
            "retrieved evidence. Cite evidence with bracketed citation_id values such "
            "as [E1] when relying on it. Do not treat these snippets as exhaustive."
        )

    def _fallback_report(self, selected) -> str:
        lines = [
            "The full report model is not loaded in this mode, so this is a basic "
            "summary of the wording flagged for review:",
            "",
        ]
        for p in selected[:12]:
            lines.append(f"- “{p.text.strip()[:200]}”")
        lines.append("")
        lines.append(
            "Run the backend with the full model (GPU) to get a grounded, "
            "chronological explanation of the bias and framing."
        )
        return "\n".join(lines)
