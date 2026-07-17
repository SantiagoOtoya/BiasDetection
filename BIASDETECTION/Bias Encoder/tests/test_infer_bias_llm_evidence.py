from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = PROJECT_ROOT / "LLM-inference"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))

import evidence_retrieval


def load_infer_module() -> types.ModuleType:
    module_path = LLM_DIR / "infer_bias_llm.py"
    spec = importlib.util.spec_from_file_location("infer_bias_llm", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load infer_bias_llm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["infer_bias_llm"] = module
    spec.loader.exec_module(module)
    return module


infer_bias_llm = load_infer_module()


class ClaimExtractingLlm:
    def generate_with_system(self, system_prompt: str, prompt: str) -> str:
        if "extract" in system_prompt.casefold():
            return (
                '[{"sentence_index": 0, "verifiability": "verifiable", '
                '"claims": ["Federal Reserve data report a wealth distribution."]}]'
            )
        if "assess" in system_prompt.casefold():
            return "[]"
        return "Report."

    def generate(self, prompt: str) -> str:
        return "Report."


class NonVerifiableLlm(ClaimExtractingLlm):
    def generate_with_system(self, system_prompt: str, prompt: str) -> str:
        if "extract" in system_prompt.casefold():
            return '[{"sentence_index": 0, "verifiability": "not_verifiable", "claims": []}]'
        return super().generate_with_system(system_prompt, prompt)


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
    def test_production_defaults_use_promoted_v3_and_pinned_llama(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["infer_bias_llm.py", "--article-text", "Example text."],
        ):
            args = infer_bias_llm.parse_args()

        self.assertEqual(
            args.sbert_model_dir,
            PROJECT_ROOT / "models" / "all-mpnet-base-v2-babe-v3" / "best",
        )
        self.assertEqual(args.selection_mode, "calibrated")
        self.assertEqual(args.llm_model, infer_bias_llm.DEFAULT_LLM_MODEL)
        self.assertEqual(args.llm_revision, infer_bias_llm.DEFAULT_LLM_REVISION)

    def test_llama_generator_propagates_pinned_revision(self) -> None:
        tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token=None)
        model = mock.Mock()
        tokenizer_loader = mock.Mock(return_value=tokenizer)
        model_loader = mock.Mock(return_value=model)
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=tokenizer_loader
        )
        fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=model_loader
        )
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            float32="float32",
        )

        with mock.patch.object(infer_bias_llm.training, "torch", fake_torch):
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                generator = infer_bias_llm.LlamaGenerator(
                    model_name="model-id",
                    hf_token="token",
                    allow_cpu=True,
                    max_new_tokens=16,
                    temperature=0.0,
                    top_p=1.0,
                )

        self.assertEqual(generator.revision, infer_bias_llm.DEFAULT_LLM_REVISION)
        tokenizer_loader.assert_called_once_with(
            "model-id",
            token="token",
            revision=infer_bias_llm.DEFAULT_LLM_REVISION,
        )
        model_loader.assert_called_once_with(
            "model-id",
            token="token",
            revision=infer_bias_llm.DEFAULT_LLM_REVISION,
            torch_dtype="float32",
        )
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()

    def test_llama_generator_accepts_chat_template_batch_encoding(self) -> None:
        class FakeTensor:
            shape = (1, 2)

            def to(self, device: str) -> "FakeTensor":
                self.device = device
                return self

        input_ids = FakeTensor()
        attention_mask = FakeTensor()
        tokenizer = types.SimpleNamespace(
            apply_chat_template=mock.Mock(
                return_value={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            ),
            eos_token_id=2,
            decode=mock.Mock(return_value="Generated report."),
        )
        model = mock.Mock()
        model.parameters.return_value = iter([types.SimpleNamespace(device="cuda:0")])
        model.generate.return_value = [[10, 11, 12]]

        generator = object.__new__(infer_bias_llm.LlamaGenerator)
        generator.tokenizer = tokenizer
        generator.model = model
        generator.torch = types.SimpleNamespace(no_grad=nullcontext)
        generator.max_new_tokens = 16
        generator.temperature = 0.0
        generator.top_p = 1.0

        result = generator.generate_with_system("System", "Prompt")

        self.assertEqual(result, "Generated report.")
        self.assertEqual(input_ids.device, "cuda:0")
        self.assertEqual(attention_mask.device, "cuda:0")
        generate_kwargs = model.generate.call_args.kwargs
        self.assertIs(generate_kwargs["input_ids"], input_ids)
        self.assertIs(generate_kwargs["attention_mask"], attention_mask)
        tokenizer.decode.assert_called_once_with([12], skip_special_tokens=True)

    def test_resolve_evidence_mode_preserves_legacy_alias(self) -> None:
        self.assertEqual(
            infer_bias_llm.resolve_evidence_mode(
                types.SimpleNamespace(enable_evidence=False)
            ),
            "off",
        )
        self.assertEqual(
            infer_bias_llm.resolve_evidence_mode(
                types.SimpleNamespace(enable_evidence=True)
            ),
            "web",
        )
        self.assertEqual(
            infer_bias_llm.resolve_evidence_mode(
                types.SimpleNamespace(enable_evidence=False, evidence_mode="web")
            ),
            "web",
        )
        self.assertEqual(
            infer_bias_llm.resolve_evidence_mode(
                types.SimpleNamespace(enable_evidence=False, evidence_mode="corpus")
            ),
            "corpus",
        )
        self.assertEqual(
            infer_bias_llm.resolve_evidence_mode(
                types.SimpleNamespace(enable_evidence=False, evidence_mode="hybrid")
            ),
            "hybrid",
        )
        with self.assertRaises(ValueError):
            infer_bias_llm.resolve_evidence_mode(
                types.SimpleNamespace(enable_evidence=True, evidence_mode="off")
            )
        with self.assertRaises(ValueError):
            infer_bias_llm.resolve_evidence_mode(
                types.SimpleNamespace(enable_evidence=True, evidence_mode="hybrid")
            )

    def test_parse_args_exposes_explicit_evidence_mode(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["infer_bias_llm.py", "--article-text", "Example text.", "--evidence-mode", "web"],
        ):
            args = infer_bias_llm.parse_args()

        self.assertEqual(args.evidence_mode, "web")

    def test_parse_args_accepts_corpus_and_hybrid_modes(self) -> None:
        for mode in ("corpus", "hybrid"):
            with self.subTest(mode=mode):
                with mock.patch.object(
                    sys,
                    "argv",
                    ["infer_bias_llm.py", "--article-text", "Example text.", "--evidence-mode", mode],
                ):
                    args = infer_bias_llm.parse_args()
                self.assertEqual(args.evidence_mode, mode)

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
                selection_config=infer_bias_llm.legacy_argmax_selection_config(),
            )

        window = record["context_windows"][0]
        self.assertEqual(window["retrieval_status"], "not_requested")
        self.assertEqual(window["evidence_items"], [])
        self.assertEqual(window["evidence_metadata"]["requested_mode"], "off")
        self.assertEqual(window["evidence_metadata"]["effective_mode"], "off")

    def test_analyze_article_does_not_call_retriever_for_explicit_off_mode(self) -> None:
        prediction = selected_prediction(0, "The proposal is a reckless giveaway.")
        args = types.SimpleNamespace(
            batch_size=16,
            context_sentences=0,
            max_windows_per_article=0,
            sbert_model_dir="model",
            llm_model="llm",
            enable_evidence=False,
            evidence_mode="off",
        )

        with mock.patch.object(
            infer_bias_llm,
            "classify_sentences",
            return_value=[prediction],
        ):
            with mock.patch.object(infer_bias_llm.evidence_retrieval, "retrieve_evidence") as retrieve:
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
                    selection_config=infer_bias_llm.legacy_argmax_selection_config(),
                )

        self.assertEqual(record["context_windows"][0]["retrieval_status"], "not_requested")
        retrieve.assert_not_called()

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
        result = evidence_retrieval.EvidenceResult(
            status="found",
            items=[evidence_item()],
            requested_mode="web",
            effective_mode="web",
            elapsed_ms=12,
            request_id="a1:window:0",
            article_id="a1",
            policy_version="trusted_sources/v1",
            provider="brave",
        )

        with mock.patch.object(
            infer_bias_llm,
            "classify_sentences",
            return_value=[prediction],
        ):
            with mock.patch.object(
                infer_bias_llm.evidence_retrieval,
                "retrieve_evidence",
                return_value=result,
            ) as retrieve:
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
                    llm=ClaimExtractingLlm(),
                    selection_config=infer_bias_llm.legacy_argmax_selection_config(),
                )

        window = record["context_windows"][0]
        self.assertEqual(window["retrieval_status"], "found")
        self.assertEqual(window["evidence_items"][0]["citation_id"], "E1")
        self.assertEqual(window["evidence_items"][0]["domain"], "federalreserve.gov")
        self.assertEqual(window["evidence_metadata"]["elapsed_ms"], 12)
        self.assertEqual(window["evidence_metadata"]["provider"], "brave")
        self.assertIn("trusted_external_evidence", window["prompt"])
        claim = retrieve.call_args.kwargs["claims"][0]
        self.assertEqual(claim.claim_id, "a1:sentence:0:claim:1")
        self.assertEqual(claim.selected_sentence_indices, (0,))

    def test_analyze_article_routes_corpus_mode_through_shared_retriever(self) -> None:
        prediction = selected_prediction(0, "A claim with corpus evidence.")
        args = types.SimpleNamespace(
            batch_size=16,
            context_sentences=0,
            max_windows_per_article=0,
            sbert_model_dir="model",
            llm_model="llm",
            enable_evidence=False,
            evidence_mode="corpus",
            evidence_provider="brave",
            max_evidence_items=5,
            evidence_timeout_seconds=1,
        )
        result = evidence_retrieval.EvidenceResult(
            status="found",
            items=[evidence_item()],
            requested_mode="corpus",
            effective_mode="corpus",
            provider="qdrant",
        )

        with mock.patch.object(
            infer_bias_llm,
            "classify_sentences",
            return_value=[prediction],
        ):
            with mock.patch.object(
                infer_bias_llm.evidence_retrieval,
                "retrieve_evidence",
                return_value=result,
            ) as retrieve:
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
                    llm=ClaimExtractingLlm(),
                    selection_config=infer_bias_llm.legacy_argmax_selection_config(),
                )

        self.assertEqual(record["context_windows"][0]["retrieval_status"], "found")
        self.assertEqual(retrieve.call_args.kwargs["mode"], "corpus")

    def test_non_verifiable_selected_sentence_never_calls_retriever(self) -> None:
        prediction = selected_prediction(0, "This proposal is an outrageous betrayal.")
        prediction.legacy_compatibility = False
        prediction.bias_assessment = "clear_bias"
        prediction.opinion_style = "opinionated_style"
        args = types.SimpleNamespace(
            batch_size=16,
            context_sentences=0,
            max_windows_per_article=0,
            sbert_model_dir="model",
            llm_model="llm",
            enable_evidence=False,
            evidence_mode="hybrid",
            evidence_provider="brave",
            max_evidence_items=5,
            evidence_timeout_seconds=1,
        )
        with mock.patch.object(infer_bias_llm, "classify_sentences", return_value=[prediction]):
            with mock.patch.object(infer_bias_llm.evidence_retrieval, "retrieve_evidence") as retrieve:
                record = infer_bias_llm.analyze_article(
                    article=infer_bias_llm.ArticleRecord("a1", prediction.text, {}),
                    model=None,
                    bias_head=None,
                    opinion_head=None,
                    bias_label2id={},
                    opinion_label2id={},
                    device=None,
                    args=args,
                    llm=NonVerifiableLlm(),
                    selection_config=infer_bias_llm.legacy_argmax_selection_config(),
                )
        retrieve.assert_not_called()
        self.assertEqual(
            record["selected_sentences"][0]["evidence_status"], "not_verifiable"
        )
        self.assertEqual(record["evidence"], [])


if __name__ == "__main__":
    unittest.main()
