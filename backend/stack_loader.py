"""Load exactly one inference stack (v2 or v3) per process.

The repository contains two generations of the inference code that share module
names (``infer_bias_llm``, ``evidence_retrieval``, ``finetune_all_mpnet_babe``,
``calibrate_sbert_heads``, …):

  - **v2**: the repo-root modules (legacy stack; 2-class softmax heads).
  - **v3**: ``BIASDETECTION/Bias Encoder`` + its ``LLM-inference`` subdirectory
    (production handoff; scalar sigmoid heads, strict artifact binding).

Because Python caches modules by *name*, importing both stacks in one process
would silently mix them (e.g. v3 ``infer_bias_llm`` bound to the v2
``finetune_all_mpnet_babe``), which would be subtly and fatally wrong. This
module enforces a hard rule: **one stack per process**. The first
``load_stack()`` call wins; asking for the other stack afterwards raises.

Every imported module's ``__file__`` is verified to live inside the expected
stack directory, so a path-ordering accident fails loudly instead of running
the wrong code.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_DIRS = (PROJECT_ROOT,)
V3_ROOT = PROJECT_ROOT / "BIASDETECTION" / "Bias Encoder"
V3_DIRS = (V3_ROOT, V3_ROOT / "LLM-inference")

# Module names that exist in one or both stacks. None of these may be imported
# before the stack choice is made, and none may resolve outside the chosen stack.
GUARDED_MODULES = (
    "infer_bias_llm",
    "evidence_retrieval",
    "finetune_all_mpnet_babe",
    "calibrate_sbert_heads",
    "data_pipeline",
    "artifact_manifest",
    "evidence_assessment",
    "runtime_config",
    "trusted_source_registry",
    "retrieval_store",
    "source_fetch",
    "corpus_operations",
)

_ACTIVE: SimpleNamespace | None = None


class StackIsolationError(RuntimeError):
    """A stack-mixing hazard was detected."""


def _stack_dirs(name: str) -> tuple[Path, ...]:
    if name == "v2":
        return V2_DIRS
    if name == "v3":
        if not V3_ROOT.exists():
            raise StackIsolationError(
                f"v3 stack directory not found: {V3_ROOT}. "
                "The BIASDETECTION handoff must be present to use MODEL_STACK=v3."
            )
        return V3_DIRS
    raise StackIsolationError(f"Unknown stack {name!r}; expected 'v2' or 'v3'.")


def _assert_module_in_dirs(module, dirs: tuple[Path, ...], name: str) -> None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise StackIsolationError(f"Module {name!r} has no __file__; cannot verify stack.")
    module_path = Path(module_file).resolve()
    for directory in dirs:
        try:
            module_path.relative_to(directory.resolve())
            return
        except ValueError:
            continue
    raise StackIsolationError(
        f"Module {name!r} resolved to {module_path}, outside the active stack "
        f"directories {[str(d) for d in dirs]}. Another copy shadowed the import."
    )


def active_stack_name() -> str | None:
    return _ACTIVE.name if _ACTIVE is not None else None


def load_stack(name: str) -> SimpleNamespace:
    """Import and return the requested stack.

    Returns a namespace with:
      - ``name``: "v2" | "v3"
      - ``dirs``: the stack's source directories
      - ``infer``: the stack's ``infer_bias_llm`` module
      - ``evidence_retrieval``: the stack's ``evidence_retrieval`` module
      - ``training``: the stack's ``finetune_all_mpnet_babe`` module
    """
    global _ACTIVE
    if _ACTIVE is not None:
        if _ACTIVE.name == name:
            return _ACTIVE
        raise StackIsolationError(
            f"Stack {_ACTIVE.name!r} is already active; cannot load {name!r} in the "
            "same process. Restart the server with the desired MODEL_STACK."
        )

    dirs = _stack_dirs(name)

    # No guarded module may be imported before the stack decision.
    already = [m for m in GUARDED_MODULES if m in sys.modules]
    if already:
        raise StackIsolationError(
            f"Modules {already} were imported before stack selection; "
            "load_stack() must run first."
        )

    # Put the stack's directories at the very front of sys.path, in order.
    for directory in reversed(dirs):
        entry = str(directory)
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)

    infer = importlib.import_module("infer_bias_llm")
    evidence_retrieval = importlib.import_module("evidence_retrieval")
    training = importlib.import_module("finetune_all_mpnet_babe")

    # Verify every guarded module that got imported resolves inside the stack.
    for module_name in GUARDED_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            _assert_module_in_dirs(module, dirs, module_name)

    _ACTIVE = SimpleNamespace(
        name=name,
        dirs=dirs,
        infer=infer,
        evidence_retrieval=evidence_retrieval,
        training=training,
    )
    return _ACTIVE
