"""Real analysis pipeline: SBERT classification + Llama report + fact-checker.

Reuses the existing, unmodified helpers from ``LLM-inference/infer_bias_llm.py``
(which itself imports ``finetune_all_mpnet_babe.py``) for:

  - sentence splitting
  - loading the fine-tuned SBERT + classification heads
  - per-sentence bias/opinion classification
  - selecting biased/opinionated sentences
  - building +/-2 sentence context windows
  - the media-bias system prompt and the Llama generator

New backend code adds: assembling the checkpoint from the HF repo, a single
cohesive report prompt, the 0-100 bias score, and the conservative fact-checker.

Modes:
  - "gpu":         load Llama for real report + fact-check generation (needs CUDA).
  - "cpu":         same, but allow slow CPU Llama generation.
  - "prompt-only": SBERT + scoring + fact-check fallback, no Llama (fast local test).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import fact_checker
import scoring
from model_loader import resolve_checkpoint
from pipeline import Analyzer
from schemas import AnalyzeRequest, AnalyzeResponse, SelectedSentence

LOGGER = logging.getLogger("bias_backend.real_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_INFERENCE_DIR = PROJECT_ROOT / "LLM-inference"
for path in (str(PROJECT_ROOT), str(LLM_INFERENCE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Reuse the existing inference module (unmodified).
import infer_bias_llm as infer  # noqa: E402


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
    ) -> None:
        self.mode = mode
        self.context_sentences = context_sentences
        self.max_windows = max_windows
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        infer.training.import_training_dependencies()
        torch = infer.training.torch
        assert torch is not None
        self._torch = torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("SBERT device: %s", self.device)

        model_dir, heads_path = resolve_checkpoint(assembled_dir=assembled_dir)
        (
            self.model,
            self.bias_head,
            self.opinion_head,
            self.bias_label2id,
            self.opinion_label2id,
        ) = infer.load_sbert_and_heads(model_dir, heads_path, self.device)
        self.sbert_loaded = True
        LOGGER.info("SBERT + classification heads loaded.")

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

    # --- analysis ---

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        sentences = infer.split_sentences(request.text)
        if not sentences:
            return AnalyzeResponse(
                overall_score=0.0,
                score_caption="No readable sentences were found.",
                report="No readable article text was found on this page.",
                meta={"mode": self.mode, "sentence_count": 0, "selected_count": 0},
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

        # --- score (Phase 8) ---
        signals = [
            scoring.sentence_signal(p.bias_probabilities, p.opinion_probabilities)
            for p in predictions
        ]
        score = scoring.compute_score(signals)
        level = scoring.bucket(score)
        caption = scoring.caption(score, len(selected), len(sentences))

        # --- report (Phase 7) ---
        report = self._build_report(windows, predictions_by_index, selected)
        # predictions_by_index is used by the report prompt builder.

        # --- fact-check (Phase 9) ---
        reviewed_sentences = [p.text for p in selected]
        context_text = "\n\n".join(w.text for w in windows)
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
            )
            for p in selected
        ]

        return AnalyzeResponse(
            overall_score=score,
            bias_level=level,
            score_caption=caption,
            report=report,
            fact_checks=fact_checks,
            selected_sentences=selected_out,
            meta={
                "mode": self.mode,
                "sentence_count": len(sentences),
                "selected_count": len(selected),
                "window_count": len(windows),
                "llm_used": self.llm is not None,
            },
        )

    # --- report helpers ---

    def _build_report(self, windows, predictions_by_index, selected) -> str:
        if not selected:
            return (
                "No strongly biased or opinionated wording was flagged in this "
                "page. The passage reads as largely factual based on the text provided."
            )
        if self.llm is None:
            return self._fallback_report(selected)

        import llm_utils

        prompt = self._build_combined_prompt(windows, predictions_by_index)
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

    def _build_combined_prompt(self, windows, predictions_by_index) -> str:
        """One prompt covering all reviewed sentences + context, in article order.

        Produces a single cohesive report instead of one report per window.
        """
        import json

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
        return (
            "Analyze only the article passage below. The sentences listed under "
            "sentences_for_review are the article sentences that need focused "
            "analysis. Use surrounding_article_context only to interpret those "
            "sentences in their original chronological order.\n\n"
            f"sentences_for_review:\n{reviewed_json}\n\n"
            f"surrounding_article_context:\n{context_block}"
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
