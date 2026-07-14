"""Hash-bound artifact manifests for later calibration and promotion sessions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


ARTIFACT_MANIFEST_SCHEMA_VERSION = "artifact_manifest/v1"


class ArtifactManifestValidationError(ValueError):
    """Raised when an artifact manifest or its declared contents are invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path, *, excluded_relative_paths: Iterable[str] = ()) -> str:
    """Hash a file or deterministic tree, optionally excluding mutable siblings."""

    path = path.resolve()
    excluded = {item.replace("\\", "/") for item in excluded_relative_paths}
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    for candidate in sorted(files, key=lambda item: item.relative_to(path).as_posix()):
        relative = candidate.relative_to(path).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(candidate).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _relative_entry(root: Path, path: Path, *, excluded_relative_paths: Iterable[str] = ()) -> dict[str, Any]:
    root = root.resolve()
    path = path.resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Artifact path must be under {root}: {path}") from exc
    entry: dict[str, Any] = {
        "path": relative,
        "kind": "directory" if path.is_dir() else "file",
        "sha256": sha256_path(path, excluded_relative_paths=excluded_relative_paths),
    }
    excluded = sorted({item.replace("\\", "/") for item in excluded_relative_paths})
    if excluded:
        entry["excluded_relative_paths"] = excluded
    return entry


def _core_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(manifest.get("core", {})))


def build_artifact_manifest(
    artifact_root: Path,
    *,
    model_path: Path,
    heads_path: Path,
    split_manifest_path: Path,
    canonical_data_manifest_path: Path,
    evaluation_scorecard_path: Path | None,
    code_paths: Iterable[Path],
    label_mappings: Mapping[str, Mapping[str, int]],
    head_type: str,
    head_architecture: Mapping[str, Any],
    training_configuration: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    model_excluded_relative_paths: Iterable[str] = (),
    calibration_path: Path | None = None,
    release_gate_results: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest with a non-circular immutable artifact identity.

    ``artifact_manifest_sha256`` is the SHA-256 of ``core`` only.  Calibration
    may safely bind it, while the final document can subsequently include the
    calibration file hash and promotion result without changing that identity.
    """

    artifact_root = artifact_root.resolve()
    code_entries = [
        _relative_entry(artifact_root, path)
        for path in sorted({Path(path).resolve() for path in code_paths}, key=lambda item: str(item))
    ]
    files: dict[str, Any] = {
        "model": _relative_entry(
            artifact_root,
            model_path,
            excluded_relative_paths=model_excluded_relative_paths,
        ),
        "classification_heads": _relative_entry(artifact_root, heads_path),
        "split_manifest": _relative_entry(artifact_root, split_manifest_path),
        "canonical_data_manifest": _relative_entry(artifact_root, canonical_data_manifest_path),
        "code": code_entries,
    }
    if evaluation_scorecard_path is not None and evaluation_scorecard_path.exists():
        files["evaluation_scorecard"] = _relative_entry(artifact_root, evaluation_scorecard_path)
    core = {
        "files": files,
        "head_type": head_type,
        "label_mappings": copy.deepcopy(dict(label_mappings)),
        "head_architecture": copy.deepcopy(dict(head_architecture)),
        "training_configuration": copy.deepcopy(dict(training_configuration)),
        "source_summary": copy.deepcopy(dict(source_summary)),
    }
    identity = canonical_json_sha256(core)
    calibration = None
    if calibration_path is not None and calibration_path.exists():
        calibration = _relative_entry(artifact_root, calibration_path)
    promotion_state = "pending_calibration" if calibration is None else "rejected"
    if release_gate_results is not None:
        promotion_state = str(
            release_gate_results.get("acceptance_gate", {}).get(
                "promotion_state", promotion_state
            )
        )
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "artifact_manifest_sha256": identity,
        "core": core,
        "calibration": calibration,
        "release_gate_results": copy.deepcopy(dict(release_gate_results))
        if release_gate_results is not None
        else None,
        "promotion_state": promotion_state,
    }
    manifest["document_sha256"] = canonical_json_sha256(manifest)
    return manifest


def validate_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    artifact_root: Path | None = None,
    verify_files: bool = False,
) -> None:
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ArtifactManifestValidationError("Unsupported artifact manifest schema version.")
    core = _core_payload(manifest)
    if not core:
        raise ArtifactManifestValidationError("Artifact manifest has no core payload.")
    expected_identity = canonical_json_sha256(core)
    if manifest.get("artifact_manifest_sha256") != expected_identity:
        raise ArtifactManifestValidationError("artifact_manifest_sha256 does not match core payload.")
    copy_without_document = copy.deepcopy(dict(manifest))
    document_sha256 = copy_without_document.pop("document_sha256", None)
    if document_sha256 and document_sha256 != canonical_json_sha256(copy_without_document):
        raise ArtifactManifestValidationError("document_sha256 does not match artifact manifest content.")
    if not verify_files:
        return
    if artifact_root is None:
        raise ValueError("artifact_root is required when verify_files=True.")
    root = artifact_root.resolve()
    file_entries = dict(core.get("files", {}))
    entries: list[Mapping[str, Any]] = []
    for key, entry in file_entries.items():
        if key == "code":
            entries.extend(entry)
        else:
            entries.append(entry)
    if manifest.get("calibration"):
        entries.append(manifest["calibration"])
    if manifest.get("locked_test_evaluation"):
        entries.append(manifest["locked_test_evaluation"])
    for entry in entries:
        path = (root / str(entry["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ArtifactManifestValidationError("Artifact entry escapes artifact root.") from exc
        if not path.exists():
            raise ArtifactManifestValidationError(f"Artifact file is missing: {entry['path']}")
        observed = sha256_path(path, excluded_relative_paths=entry.get("excluded_relative_paths", ()))
        if observed != entry.get("sha256"):
            raise ArtifactManifestValidationError(f"Artifact hash mismatch: {entry['path']}")


def write_artifact_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    validate_artifact_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def bind_calibration(
    manifest: Mapping[str, Any],
    *,
    artifact_root: Path,
    calibration_path: Path,
    release_gate_results: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach calibration/promotion results without changing immutable core identity."""

    validate_artifact_manifest(manifest)
    result = copy.deepcopy(dict(manifest))
    result["calibration"] = _relative_entry(artifact_root, calibration_path)
    freeze_core = {
        "schema_version": "frozen_artifact/v1",
        "artifact_manifest_sha256": result["artifact_manifest_sha256"],
        "calibration_path": result["calibration"]["path"],
        "calibration_sha256": result["calibration"]["sha256"],
    }
    result["frozen_artifact"] = {
        **freeze_core,
        "frozen_artifact_sha256": canonical_json_sha256(freeze_core),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    result["locked_test_evaluation"] = None
    result["release_gate_results"] = (
        copy.deepcopy(dict(release_gate_results))
        if release_gate_results is not None
        else None
    )
    gate_results = result.get("release_gate_results") or {}
    result["promotion_state"] = str(
        gate_results.get("acceptance_gate", {}).get(
            "promotion_state", "pending_locked_test"
        )
    )
    result.pop("document_sha256", None)
    result["document_sha256"] = canonical_json_sha256(result)
    validate_artifact_manifest(result)
    return result


def bind_locked_test_evaluation(
    manifest: Mapping[str, Any],
    *,
    artifact_root: Path,
    scorecard_path: Path,
    release_gate_results: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the sole locked-test scorecard to an already frozen artifact."""

    validate_artifact_manifest(manifest, artifact_root=artifact_root, verify_files=True)
    if manifest.get("promotion_state") != "pending_locked_test":
        raise ArtifactManifestValidationError(
            "Locked-test evaluation requires promotion_state='pending_locked_test'."
        )
    if not manifest.get("frozen_artifact"):
        raise ArtifactManifestValidationError("Artifact has no frozen_artifact identity.")
    if manifest.get("locked_test_evaluation") is not None:
        raise ArtifactManifestValidationError("Artifact already has a locked-test evaluation.")
    gate = dict(release_gate_results).get("acceptance_gate", {})
    promotion_state = str(gate.get("promotion_state", "rejected"))
    if promotion_state not in {"eligible", "rejected"}:
        raise ArtifactManifestValidationError("Invalid locked-test promotion state.")
    result = copy.deepcopy(dict(manifest))
    result["locked_test_evaluation"] = _relative_entry(artifact_root, scorecard_path)
    result["release_gate_results"] = copy.deepcopy(dict(release_gate_results))
    result["promotion_state"] = promotion_state
    result.pop("document_sha256", None)
    result["document_sha256"] = canonical_json_sha256(result)
    validate_artifact_manifest(result, artifact_root=artifact_root, verify_files=True)
    return result


def refresh_artifact_core(manifest: Mapping[str, Any], *, artifact_root: Path) -> dict[str, Any]:
    """Rehash mutable declared files before binding a new calibration result."""

    validate_artifact_manifest(manifest)
    result = copy.deepcopy(dict(manifest))
    root = artifact_root.resolve()
    files = result["core"]["files"]
    entries: list[Mapping[str, Any]] = []
    for key, entry in files.items():
        if key == "code":
            entries.extend(entry)
        else:
            entries.append(entry)
    for entry in entries:
        path = (root / str(entry["path"])).resolve()
        entry["sha256"] = sha256_path(
            path,
            excluded_relative_paths=entry.get("excluded_relative_paths", ()),
        )
    result["artifact_manifest_sha256"] = canonical_json_sha256(result["core"])
    result["calibration"] = None
    result["frozen_artifact"] = None
    result["locked_test_evaluation"] = None
    result["release_gate_results"] = None
    result["promotion_state"] = "pending_calibration"
    result.pop("document_sha256", None)
    result["document_sha256"] = canonical_json_sha256(result)
    validate_artifact_manifest(result)
    return result


def load_artifact_manifest(path: Path, *, artifact_root: Path | None = None, verify_files: bool = False) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_artifact_manifest(
        manifest,
        artifact_root=artifact_root,
        verify_files=verify_files,
    )
    return manifest
