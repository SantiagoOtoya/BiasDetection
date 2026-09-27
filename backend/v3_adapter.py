"""V3 analyzer: serves the production v3 stack through the backend API.

Wraps the v3 handoff's own ``analyze_article`` (strict calibrated selection,
claim extraction, trusted-evidence assessment, citation-guarded report) rather
than re-implementing it, and maps the resulting record onto the extension's
``AnalyzeResponse`` schema.

Behavioral decisions (approved):
  - selection: strict calibrated, ``clear_bias`` only; ``INCLUDE_POSSIBLE_BIAS``
    env opt-in adds ``possible_bias``;
  - evidence: web mode (Brave) only, and only when ``ENABLE_EVIDENCE=true`` AND
    the required env vars are present — otherwise evidence mode "off" with the
    reason surfaced in ``meta``; corpus/Qdrant stays unwired;
  - fact-check statuses: supported→verified, contradicted→disputed,
    insufficient/not_verifiable→unverified — never "false";
  - the v2 stack remains the default; this analyzer only runs when
    ``MODEL_STACK=v3``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

# Stack selection MUST happen before any guarded module import.
import stack_loader

_STACK = stack_loader.load_stack("v3")
infer = _STACK.infer
evidence_retrieval = _STACK.evidence_retrieval
training = _STACK.training

import relevance_filter
import v3_mapping
from model_loader import resolve_v3_checkpoint
from pipeline import Analyzer
from schemas import AnalyzeRequest, AnalyzeResponse

LOGGER = logging.getLogger("bias_backend.v3_adapter")


class V3Analyzer(Analyzer):
    def __init__(
        self,
        mode: str = "gpu",
        context_sentences: int = 2,
        max_windows: int = 0,
        max_new_tokens: int = 700,
        temperature: float = 0.2,
        top_p: float = 0.9,
        enable_evidence: bool = False,
        max_evidence_items: int = 5,
        evidence_timeout_seconds: float = 10.0,
        enable_relevance: bool = True,
        enable_semantic_relevance: bool = True,
        relevance_threshold: float = relevance_filter.DEFAULT_THRESHOLD,
        include_possible_bias: bool = False,
    ) -> None:
        self.mode = mode
        self.context_sentences = context_sentences
        self.max_windows = max_windows
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_relevance = enable_relevance
        self.enable_semantic_relevance = enable_semantic_relevance
        self.relevance_threshold = relevance_threshold
        self.include_possible_bias = include_possible_bias
        self.max_evidence_items = max_evidence_items
        self.evidence_timeout_seconds = evidence_timeout_seconds

        training.import_training_dependencies()
        torch = training.torch
        assert torch is not None
        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("v3 SBERT device: %s", self.device)

        # Phase C: ensures model.safetensors is present in the handoff model dir
        # (downloads from the published v3 repo on first run).
        self.model_dir, self.heads_path = resolve_v3_checkpoint()

        (
            self.model,
            self.bias_head,
            self.opinion_head,
            self.bias_label2id,
            self.opinion_label2id,
        ) = infer.load_sbert_and_heads(
            self.model_dir, self.heads_path, self.device, compatibility_mode="strict"
        )
        LOGGER.info("v3 SBERT + scalar heads loaded (strict mode).")

        # Strict calibrated selection with full artifact binding. Any hash or
        # schema mismatch raises here — the server refuses to start on v3
        # rather than silently running the wrong model.
        self.selection_config = self._resolve_selection_config()
        self.sbert_loaded = True
        LOGGER.info(
            "v3 selection: mode=%s include_possible_bias=%s",
            self.selection_config.effective_mode,
            self.include_possible_bias,
        )

        # Evidence: web-only, gated on the toggle AND runtime readiness.
        self.evidence_mode = "off"
        self.evidence_unavailable_reason: str | None = None
        self.evidence_policy = evidence_retrieval.DEFAULT_TRUSTED_SOURCE_POLICY
        if enable_evidence:
            import runtime_config

            readiness = runtime_config.check_runtime_readiness("web")
            if readiness.ready:
                self.evidence_mode = "web"
                self.evidence_policy = evidence_retrieval.load_trusted_source_policy(None)
                LOGGER.info("v3 trusted web evidence enabled.")
            else:
                self.evidence_unavailable_reason = (
                    "missing env: " + ", ".join(readiness.missing_keys)
                )
                LOGGER.warning(
                    "Evidence requested but unavailable (%s); continuing with mode=off.",
                    self.evidence_unavailable_reason,
                )

        self.llm = None
        if mode in ("gpu", "cpu"):
            self.llm = infer.LlamaGenerator(
                model_name=infer.DEFAULT_LLM_MODEL,
                hf_token=None,
                allow_cpu=(mode == "cpu"),
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                revision=infer.DEFAULT_LLM_REVISION,
            )
            self.llm_loaded = True
            LOGGER.info(
                "v3 Llama loaded (%s @ %s).",
                infer.DEFAULT_LLM_MODEL,
                infer.DEFAULT_LLM_REVISION[:12],
            )
        else:
            LOGGER.info("v3 prompt-only mode: skipping Llama load.")

    def _resolve_selection_config(self):
        args = SimpleNamespace(
            selection_mode="calibrated",
            sbert_model_dir=self.model_dir,
            calibration_file=None,
            artifact_manifest=None,
            classification_heads=None,
            bias_threshold=None,
            opinion_threshold=None,
            include_possible_bias=self.include_possible_bias,
        )
        return infer.resolve_selection_config(
            args,
            self.bias_label2id,
            self.opinion_label2id,
            getattr(self.opinion_head, "head_type", None),
            heads_path=self.heads_path,
            checkpoint_head_type_declared=getattr(
                self.opinion_head, "checkpoint_declared_head_type", None
            ),
            bias_head_type=getattr(self.bias_head, "head_type", None),
            classifier_head_type=getattr(self.opinion_head, "classifier_head_type", None),
            checkpoint_schema_version=getattr(
                self.opinion_head, "checkpoint_schema_version", None
            ),
            opinion_style_target_mapping=getattr(
                self.opinion_head, "opinion_style_target_mapping", None
            ),
            classifier_input_contract=getattr(
                self.opinion_head, "classifier_input_contract", None
            ),
        )

    def _embed_texts(self, texts: list[str]):
        """Raw-text embeddings for the relevance filter (no [TARGET] marking —
        topical similarity, not classification)."""
        torch = self._torch
        with torch.no_grad():
            return training.sentence_embeddings(self.model, texts, self.device).detach().cpu()

    # --- analysis ---

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        raw_sentences = relevance_filter.split_into_paragraph_sentences(
            request.text, infer.split_sentences
        )
        if not raw_sentences:
            return self._empty_response("No readable sentences were found.")

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
            return self._empty_response(
                "No relevant article sentences were found.", relevance_meta
            )

        article = infer.ArticleRecord(
            article_id=request.url or "extension-article",
            text=" ".join(sentences),
            metadata={"source": "extension", "title": request.title or ""},
        )
        args = SimpleNamespace(
            batch_size=16,
            context_sentences=self.context_sentences,
            max_windows_per_article=(request.max_windows or self.max_windows),
            sbert_model_dir=self.model_dir,
            llm_model=infer.DEFAULT_LLM_MODEL,
            llm_revision=infer.DEFAULT_LLM_REVISION,
            evidence_mode=self.evidence_mode,
            evidence_provider="brave",
            max_evidence_items=self.max_evidence_items,
            evidence_timeout_seconds=self.evidence_timeout_seconds,
            trusted_sources_file=None,
            include_all_sentence_predictions=True,
            defer_report_generation=False,
        )
        record = infer.analyze_article(
            article=article,
            model=self.model,
            bias_head=self.bias_head,
            opinion_head=self.opinion_head,
            bias_label2id=self.bias_label2id,
            opinion_label2id=self.opinion_label2id,
            device=self.device,
            args=args,
            llm=self.llm,
            evidence_policy=self.evidence_policy,
            selection_config=self.selection_config,
        )
        return v3_mapping.record_to_response(
            record,
            relevance_meta,
            mode=self.mode,
            llm_used=self.llm is not None,
            evidence_mode=self.evidence_mode,
            evidence_unavailable_reason=self.evidence_unavailable_reason,
        )

    def _empty_response(self, caption: str, relevance_meta: dict | None = None) -> AnalyzeResponse:
        meta = {
            "mode": self.mode,
            "stack": "v3",
            "sentence_count": 0,
            "selected_count": 0,
        }
        if relevance_meta is not None:
            meta["relevance"] = relevance_meta
        return AnalyzeResponse(
            overall_score=0.0,
            score_caption=caption,
            report="No readable article text was found on this page.",
            meta=meta,
        )
