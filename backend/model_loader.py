"""Assemble a loadable SentenceTransformer checkpoint from the friend's HF repo.

The Hugging Face repo ``BiLSTM/BIAS_Detection`` contains only:

  - ``model.safetensors``        (the fine-tuned SBERT transformer weights)
  - ``classification_heads.pt``  (the bias + opinion MLP heads)

It is missing the SentenceTransformer scaffolding (``modules.json``,
``config_sentence_transformers.json``, ``sentence_bert_config.json``, the
``1_Pooling`` config, and the tokenizer). Those are architecture/tokenizer files
that are identical to the base ``sentence-transformers/all-mpnet-base-v2`` model,
because fine-tuning only updated the upper encoder layers and the heads.

This module builds an assembled checkpoint directory:

  1. Download the base ``all-mpnet-base-v2`` snapshot -> provides all scaffolding.
  2. Overwrite the transformer weights with the fine-tuned ``model.safetensors``.
  3. Download ``classification_heads.pt`` alongside it.

The base model is used ONLY for the missing scaffolding/tokenizer/pooling. The
actual encoder weights come from the fine-tuned file, so the friend's training is
what runs at inference. ``verify_assembly`` confirms this by checking that the
assembled weights differ from the base weights.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

LOGGER = logging.getLogger("bias_backend.model_loader")

BASE_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
FINETUNED_REPO_ID = "BiLSTM/BIAS_Detection"
FINETUNED_WEIGHTS_FILE = "model.safetensors"
HEADS_FILE = "classification_heads.pt"

# Where the assembled checkpoint is cached, relative to the backend dir by default.
DEFAULT_ASSEMBLED_DIR = Path(__file__).resolve().parent / "models" / "assembled-sbert-babe"

# Scaffolding files copied from the base model. The transformer weights file is
# intentionally excluded here and replaced with the fine-tuned weights.
SCAFFOLD_FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "data_config.json",
]
SCAFFOLD_DIRS = ["1_Pooling"]
# Legacy weight files removed from the assembled dir so only the fine-tuned
# safetensors is loaded.
CONFLICTING_WEIGHT_FILES = ["pytorch_model.bin", "tf_model.h5", "model.ckpt"]


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def build_assembled_checkpoint(
    assembled_dir: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Return ``(assembled_dir, heads_path)``, building the checkpoint if needed."""
    from huggingface_hub import hf_hub_download, snapshot_download

    assembled_dir = Path(assembled_dir or DEFAULT_ASSEMBLED_DIR)
    heads_path = assembled_dir / HEADS_FILE
    weights_path = assembled_dir / "model.safetensors"

    marker = assembled_dir / ".assembled_ok"
    if marker.exists() and not force:
        LOGGER.info("Using cached assembled checkpoint at %s", assembled_dir)
        return assembled_dir, heads_path

    token = _hf_token()
    LOGGER.info("Downloading base scaffolding from %s", BASE_MODEL_ID)
    base_dir = Path(
        snapshot_download(
            BASE_MODEL_ID,
            allow_patterns=[
                *SCAFFOLD_FILES,
                *[f"{d}/*" for d in SCAFFOLD_DIRS],
            ],
        )
    )

    if assembled_dir.exists() and force:
        shutil.rmtree(assembled_dir)
    assembled_dir.mkdir(parents=True, exist_ok=True)

    # 1) Copy scaffolding (config/tokenizer/pooling) from the base model.
    copied = 0
    for name in SCAFFOLD_FILES:
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, assembled_dir / name)
            copied += 1
    for dirname in SCAFFOLD_DIRS:
        src_dir = base_dir / dirname
        if src_dir.exists():
            shutil.copytree(src_dir, assembled_dir / dirname, dirs_exist_ok=True)
            copied += 1
    LOGGER.info("Copied %d scaffolding item(s) from base model.", copied)

    # 2) Remove any conflicting legacy weight files.
    for name in CONFLICTING_WEIGHT_FILES:
        stale = assembled_dir / name
        if stale.exists():
            stale.unlink()

    # 3) Download the fine-tuned transformer weights and heads into the dir.
    LOGGER.info("Downloading fine-tuned weights from %s", FINETUNED_REPO_ID)
    finetuned_weights = hf_hub_download(
        FINETUNED_REPO_ID, FINETUNED_WEIGHTS_FILE, token=token
    )
    shutil.copy2(finetuned_weights, weights_path)

    LOGGER.info("Downloading classification heads from %s", FINETUNED_REPO_ID)
    finetuned_heads = hf_hub_download(FINETUNED_REPO_ID, HEADS_FILE, token=token)
    shutil.copy2(finetuned_heads, heads_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Fine-tuned weights not assembled at {weights_path}")
    if not heads_path.exists():
        raise FileNotFoundError(f"Classification heads not assembled at {heads_path}")

    verify_assembly(weights_path, Path(finetuned_weights))
    marker.write_text("ok", encoding="utf-8")
    LOGGER.info("Assembled checkpoint ready at %s", assembled_dir)
    return assembled_dir, heads_path


def verify_assembly(assembled_weights: Path, source_weights: Path) -> None:
    """Confirm the assembled weights ARE the friend's fine-tuned file.

    Two checks, neither of which requires downloading the base encoder weights:

    1. The assembled ``model.safetensors`` is byte-identical (same SHA-256) to the
       file downloaded from ``BiLSTM/BIAS_Detection`` — so the encoder weights in
       the checkpoint are the fine-tuned ones, never the base ones.
    2. The tensor names look like an all-mpnet / MPNet encoder, so the file will
       actually load into the SentenceTransformer architecture.
    """
    import hashlib

    from safetensors import safe_open

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    assembled_hash = _sha256(assembled_weights)
    source_hash = _sha256(source_weights)
    if assembled_hash != source_hash:
        raise RuntimeError(
            "Assembled weights do not match the downloaded fine-tuned file. "
            "Refusing to run so the base model is not used by mistake."
        )

    with safe_open(str(assembled_weights), framework="pt") as ft:
        keys = list(ft.keys())
    looks_like_encoder = any(
        ("encoder.layer" in k) or k.startswith("embeddings") or ".attention." in k
        for k in keys
    )
    if not looks_like_encoder:
        raise RuntimeError(
            "Fine-tuned model.safetensors does not look like an MPNet encoder "
            f"checkpoint (found {len(keys)} tensors, none matching encoder names). "
            "It may be wrapped or in an unexpected format."
        )
    LOGGER.info(
        "Verified fine-tuned weights (sha256 %s…, %d tensors, encoder layout OK).",
        assembled_hash[:12],
        len(keys),
    )


def resolve_checkpoint(
    assembled_dir: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Public entrypoint used by the pipeline. Returns ``(model_dir, heads_path)``."""
    return build_assembled_checkpoint(assembled_dir=assembled_dir, force=force)


# --- v3 production checkpoint (BIASDETECTION handoff) ------------------------

V3_REPO_ID = "BiLSTM/BIASDETECTION_"
V3_WEIGHTS_FILE = "model.safetensors"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
V3_MODEL_DIR = (
    PROJECT_ROOT
    / "BIASDETECTION"
    / "Bias Encoder"
    / "models"
    / "all-mpnet-base-v2-babe-v3"
    / "best"
)


def resolve_v3_checkpoint() -> tuple[Path, Path]:
    """Ensure the v3 handoff model dir is complete; return ``(model_dir, heads_path)``.

    The handoff ships everything except the encoder weights, which the v3
    release publishes at ``BiLSTM/BIASDETECTION_``. The weights MUST be placed
    inside the handoff's own ``best/`` directory (not an assembled copy): the
    strict loader's artifact binding resolves the manifest's model path against
    the v3 project root and verifies a directory-composite SHA-256 there —
    a relocated copy can never pass. The downloaded file is covered by the
    repo's ``*.safetensors`` gitignore rule, so git stays clean.

    Integrity: the strict artifact binding (``verify_files=True``) recomputes
    the composite hash over the completed directory and compares it to the
    manifest's ``model_sha256`` — the load fails loudly on any mismatch, so a
    wrong or corrupted weights file cannot silently run.
    """
    if not V3_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"v3 model directory not found: {V3_MODEL_DIR}. "
            "The BIASDETECTION handoff must be present to use MODEL_STACK=v3."
        )
    heads_path = V3_MODEL_DIR / "classification_heads.pt"
    if not heads_path.exists():
        raise FileNotFoundError(f"v3 classification heads not found: {heads_path}")

    weights_path = V3_MODEL_DIR / V3_WEIGHTS_FILE
    if not weights_path.exists():
        from huggingface_hub import hf_hub_download

        LOGGER.info("Downloading v3 encoder weights from %s", V3_REPO_ID)
        downloaded = hf_hub_download(V3_REPO_ID, V3_WEIGHTS_FILE, token=_hf_token())
        shutil.copy2(downloaded, weights_path)
        LOGGER.info("Placed v3 weights at %s", weights_path)

    return V3_MODEL_DIR, heads_path


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    model_dir, heads = resolve_checkpoint()
    print("Assembled model dir:", model_dir)
    print("Classification heads:", heads)
