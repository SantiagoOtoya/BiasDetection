# BIAS Detection Encoder Handoff

Created from the local workspace on 2026-07-13. This is a sanitized,
encoder-only source handoff. It is not a Git clone, branch, fork, commit, or
upload.

## Included

- Current bias-encoder training, calibration, evaluation, inference,
  retrieval, corpus-operation, and evidence-assessment source.
- Encoder unit tests, requirements, policies, fixtures, manifests, and current
  operational/release documentation.
- Production-v3 data/split manifests and locked-test release records.
- Lightweight promoted-v3 model metadata and `classification_heads.pt`.

## Deliberately excluded

- The complete `Backend` and `ExtensionUI` trees.
- Environment and credential files of every kind.
- Virtual environments, Git/agent metadata, caches, logs, and temporary files.
- Training datasets, downloaded corpora, local Qdrant data, and generated
  inference outputs.
- Encoder `model.safetensors`, historical checkpoints, and all Llama weights.

The local Llama runtime is pinned to
`meta-llama/Llama-3.1-8B-Instruct` revision
`0e9e39f249a16976918f6564b8830bc894c89659`, but its Hugging Face cache is not
part of this handoff.

## Validation

`HANDOFF_INVENTORY.json` records the payload files, sizes, and hashes.
`SHA256SUMS.txt` provides a portable integrity list. The handoff is accepted
only after Python compilation and all encoder unit tests pass from this copy.

From CMD, after creating and activating a compatible Python environment, the
unit suite can be run with:

```cmd
python -m unittest discover -s "Bias Encoder\tests" -p "test_*.py"
```

Production inference additionally requires the excluded promoted encoder
weight file and the separately cached, licensed Llama snapshot.
